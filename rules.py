"""Rule engine: explains WHY a policy looks risky and what to change.
This layer is deliberately not ML. ML says how risky; rules say why."""
from features import (parse_policy, allow_statements, actions_of, resources_of,
                      is_wildcard_action, PRIV_ESC_ACTIONS, SENSITIVE_DATA_ACTIONS)

def explain(policy):
    """Return (risk_factors, recommendations) as lists of strings."""
    policy = parse_policy(policy)
    allows = allow_statements(policy)
    factors, recs = [], []

    all_actions = [a for s in allows for a in actions_of(s)]
    lower = {a.lower() for a in all_actions}

    if "*" in all_actions:
        factors.append('Full wildcard action ("*") grants every AWS API call.')
        recs.append('Replace "*" with the specific actions the principal actually needs.')
    svc_wild = sorted({a for a in all_actions if a != "*" and is_wildcard_action(a)})
    if svc_wild:
        factors.append("Service-level wildcard actions: " + ", ".join(svc_wild))
        recs.append("Expand service wildcards (e.g. s3:*) into the individual actions required.")
    if any("NotAction" in s for s in allows):
        factors.append("Allow + NotAction grants everything except the listed actions.")
        recs.append("Use an explicit Action list instead of NotAction on Allow statements.")

    wild_res = any(r == "*" for s in allows for r in resources_of(s))
    if wild_res:
        factors.append('Wildcard resource ("*") applies the permissions to every resource.')
        recs.append("Restrict Resource to specific ARNs.")

    iam_acts = sorted(a for a in all_actions if a.lower().startswith("iam:"))
    if iam_acts:
        factors.append("IAM permissions present: " + ", ".join(iam_acts[:6]) + (" ..." if len(iam_acts) > 6 else ""))
        recs.append("Limit IAM permissions to the minimum, and scope them to specific roles/users.")

    esc = sorted(a for a in all_actions if a.lower() in PRIV_ESC_ACTIONS)
    if esc:
        factors.append("Known privilege-escalation actions: " + ", ".join(esc))
        recs.append("Add conditions (e.g. iam:PassedToService, permissions boundaries) around escalation-capable actions.")

    sens = sorted(a for a in all_actions if a.lower() in SENSITIVE_DATA_ACTIONS)
    if sens:
        factors.append("Sensitive data / audit-control actions: " + ", ".join(sens))
        recs.append("Scope secrets, KMS and logging permissions to specific keys, secrets and trails.")

    sts = sorted(a for a in all_actions if a.lower().startswith("sts:"))
    if sts:
        factors.append("STS permissions present: " + ", ".join(sts))
        recs.append("Restrict AssumeRole to specific role ARNs and require conditions such as MFA.")

    if len(allows) >= 2 and len({a.split(":")[0].lower() for a in all_actions if ":" in a}) >= 3:
        factors.append("Spans several AWS services in one policy.")
        recs.append("Split into smaller, purpose-specific policies.")

    has_cond = any(s.get("Condition") for s in allows)
    if (wild_res or "*" in all_actions) and not has_cond:
        factors.append("No Condition block narrows when the broad permissions apply.")
        recs.append("Add conditions (source IP, MFA, tags, region) to limit when access is allowed.")

    if not factors:
        factors.append("No obvious risk indicators found.")
        recs.append("Policy looks narrowly scoped. Keep reviewing it as permissions evolve.")

    # de-duplicate, keep order
    recs = list(dict.fromkeys(recs))
    return factors, recs
