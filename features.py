"""Feature extraction: IAM policy JSON -> 8 numeric features."""
import json

FEATURE_NAMES = [
    "statement_count",
    "allow_count",
    "wildcard_action",
    "wildcard_resource",
    "service_count",
    "iam_action_count",
    "sts_action_count",
    "has_condition",
    # added to separate read-only policies from ones that can change/destroy things
    "write_action_count",
    "delete_action_count",
    "escalation_action_count",
    "sensitive_action_count",
]

# Actions widely documented as privilege-escalation paths or data/audit exposure risks.
PRIV_ESC_ACTIONS = {
    "iam:passrole", "iam:createaccesskey", "iam:createloginprofile",
    "iam:updateloginprofile", "iam:attachuserpolicy", "iam:attachrolepolicy",
    "iam:attachgrouppolicy", "iam:putuserpolicy", "iam:putrolepolicy",
    "iam:putgrouppolicy", "iam:createpolicyversion", "iam:setdefaultpolicyversion",
    "iam:updateassumerolepolicy", "iam:addusertogroup", "iam:createrole",
    "iam:createuser", "sts:assumerole",
}
SENSITIVE_DATA_ACTIONS = {
    "kms:decrypt", "kms:*", "secretsmanager:getsecretvalue",
    "ssm:getparameter", "ssm:getparameters", "s3:deletebucket",
    "cloudtrail:stoplogging", "cloudtrail:deletetrail",
}


def is_delete_action(a):
    return a.split(":")[-1].lower().startswith(
        ("delete", "terminate", "remove", "deregister", "stoplogging"))


def is_write_action(a):
    """Write-style actions that are not deletes."""
    n = a.split(":")[-1].lower()
    return (not is_delete_action(a)) and n.startswith(
        ("put", "create", "update", "start", "stop", "send", "publish",
         "invoke", "attach", "modify", "run", "write", "upload"))


class InvalidPolicy(ValueError):
    pass


def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def parse_policy(policy):
    """Accept a JSON string or dict; return a validated dict."""
    if isinstance(policy, str):
        try:
            policy = json.loads(policy)
        except json.JSONDecodeError as e:
            raise InvalidPolicy(f"JSON could not be parsed: {e}")
    if not isinstance(policy, dict):
        raise InvalidPolicy("Policy must be a JSON object.")
    if "Statement" not in policy:
        raise InvalidPolicy('Policy has no "Statement" element.')
    stmts = _as_list(policy["Statement"])
    if not stmts or not all(isinstance(s, dict) for s in stmts):
        raise InvalidPolicy('"Statement" must be an object or a non-empty list of objects.')
    for s in stmts:
        if s.get("Effect") not in ("Allow", "Deny"):
            raise InvalidPolicy('Each statement needs "Effect": "Allow" or "Deny".')
        if not (s.get("Action") or s.get("NotAction")):
            raise InvalidPolicy('Each statement needs "Action" (or "NotAction").')
    policy["Statement"] = stmts
    return policy


def allow_statements(policy):
    return [s for s in policy["Statement"] if s.get("Effect") == "Allow"]


def actions_of(stmt):
    return [str(a) for a in _as_list(stmt.get("Action"))]


def resources_of(stmt):
    return [str(r) for r in _as_list(stmt.get("Resource"))]


def is_wildcard_action(a):
    """'*' or 'service:*' counts as a wildcard action."""
    return a == "*" or a.endswith(":*")


def extract_features(policy):
    """Return dict of the 8 features. Only Allow statements drive risk;
    Deny statements are counted in statement_count only."""
    policy = parse_policy(policy)
    allows = allow_statements(policy)

    all_actions = [a for s in allows for a in actions_of(s)]
    # NotAction on an Allow is effectively "everything except" -> treat as wildcard
    not_action_allow = any("NotAction" in s for s in allows)

    wildcard_action = int(any(is_wildcard_action(a) for a in all_actions) or not_action_allow)
    wildcard_resource = int(any(r == "*" for s in allows for r in resources_of(s)))

    services = {a.split(":")[0].lower() for a in all_actions if a != "*" and ":" in a}
    iam_count = sum(1 for a in all_actions if a.lower().startswith("iam:"))
    sts_count = sum(1 for a in all_actions if a.lower().startswith("sts:"))
    has_cond = int(any(s.get("Condition") for s in allows))
    low = [a.lower() for a in all_actions]

    return {
        "statement_count": len(policy["Statement"]),
        "allow_count": len(allows),
        "wildcard_action": wildcard_action,
        "wildcard_resource": wildcard_resource,
        "service_count": len(services),
        "iam_action_count": iam_count,
        "sts_action_count": sts_count,
        "has_condition": has_cond,
        "write_action_count": sum(is_write_action(a) for a in all_actions),
        "delete_action_count": sum(is_delete_action(a) for a in all_actions),
        "escalation_action_count": sum(a in PRIV_ESC_ACTIONS for a in low),
        "sensitive_action_count": sum(a in SENSITIVE_DATA_ACTIONS for a in low),
    }


def feature_vector(policy):
    f = extract_features(policy)
    return [f[n] for n in FEATURE_NAMES]
