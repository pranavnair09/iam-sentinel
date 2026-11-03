"""Generate the labeled policy dataset (policies.csv).

Structure:  ~130 BASE policies (archetypes) x ~8 VARIATIONS each = ~1,000 policies.
Every row keeps a `group` id (its base policy) so train.py can keep all variations
of one base together in the same split (see README, "why grouped splitting").

Labels come from `rubric_label()` below, a written scoring rubric fixed BEFORE any
model is trained. It looks at things the 8 features can't see (e.g. which specific
IAM actions allow privilege escalation, delete vs. read), so the ML model is learning
an approximation of the rubric from coarse features, not copying it.
"""
import copy
import json
import random

import pandas as pd

from features import (extract_features, FEATURE_NAMES, allow_statements, actions_of, resources_of,
                      PRIV_ESC_ACTIONS, SENSITIVE_DATA_ACTIONS)

SEED = 42
N_BASES = 130
VARIATIONS = 8

# --------------------------------------------------------------------------
# Labeling rubric  (Low < 2  <= Medium < 5 <= High)
# --------------------------------------------------------------------------
def _is_delete(a):
    n = a.split(":")[-1].lower()
    return n.startswith(("delete", "terminate", "remove", "deregister", "stoplogging"))


def _is_write(a):
    n = a.split(":")[-1].lower()
    return n.startswith(("put", "create", "update", "start", "stop", "send", "publish",
                         "invoke", "attach", "modify", "run", "write", "upload"))


def statement_score(s):
    acts = actions_of(s)
    res = resources_of(s)
    low = [a.lower() for a in acts]
    score = 0.0
    broad = False

    if "*" in acts or "NotAction" in s:
        score += 5
        broad = True
    wild = {a for a in low if a != "*" and a.endswith(":*")}
    w = 0
    for a in wild:
        if a == "iam:*":
            w += 4
        elif a in ("sts:*", "kms:*", "secretsmanager:*", "cloudtrail:*"):
            w += 3
        else:
            w += 2
    score += min(w, 4)
    if wild:
        broad = True

    esc = [a for a in low if a in PRIV_ESC_ACTIONS]
    score += min(3 * len(esc), 6)
    sens = [a for a in low if a in SENSITIVE_DATA_ACTIONS and not a.endswith(":*")]
    score += min(2 * len(sens), 4)
    if esc or sens:
        broad = True

    other_iam = [a for a in low if a.startswith("iam:") and a not in PRIV_ESC_ACTIONS and not a.endswith(":*")]
    score += min(len(other_iam), 2)

    # plain write/delete operations on their own never push a policy past Medium
    rw = min(1.5 * sum(_is_delete(a) for a in acts), 3) + min(1.0 * sum(_is_write(a) and not _is_delete(a) for a in acts), 2)
    score += min(rw, 4)

    if "*" in res:
        score += 2
        if broad:
            score += 2
    if score > 0 and s.get("Condition"):
        score -= 1
    if "*" in acts or "NotAction" in s:
        score = max(score, 5)          # rubric: Action "*" is always High
    return max(score, 0.0)


def rubric_score(policy):
    allows = allow_statements(policy)
    scores = sorted((statement_score(s) for s in allows), reverse=True)
    if not scores:
        return 0.0
    total = scores[0] + min(2.0, 0.5 * sum(scores[1:]))
    services = {a.split(":")[0].lower() for s in allows for a in actions_of(s) if ":" in a}
    if len(services) >= 3:
        total += 1
    return total


def rubric_label(policy):
    sc = rubric_score(policy)
    return "High" if sc >= 5 else "Medium" if sc >= 2 else "Low"


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
READ = {
    "s3": ["s3:GetObject", "s3:ListBucket", "s3:GetBucketLocation"],
    "dynamodb": ["dynamodb:GetItem", "dynamodb:Query", "dynamodb:Scan", "dynamodb:DescribeTable"],
    "ec2": ["ec2:DescribeInstances", "ec2:DescribeVolumes", "ec2:DescribeSecurityGroups"],
    "logs": ["logs:GetLogEvents", "logs:DescribeLogGroups", "logs:FilterLogEvents"],
    "lambda": ["lambda:GetFunction", "lambda:ListFunctions"],
    "rds": ["rds:DescribeDBInstances", "rds:ListTagsForResource"],
    "sqs": ["sqs:ReceiveMessage", "sqs:GetQueueAttributes"],
    "cloudwatch": ["cloudwatch:GetMetricData", "cloudwatch:DescribeAlarms"],
}
WRITE = {
    "s3": ["s3:PutObject", "s3:PutObjectAcl"],
    "dynamodb": ["dynamodb:PutItem", "dynamodb:UpdateItem"],
    "ec2": ["ec2:StartInstances", "ec2:StopInstances", "ec2:RunInstances"],
    "logs": ["logs:PutLogEvents", "logs:CreateLogStream"],
    "lambda": ["lambda:InvokeFunction", "lambda:UpdateFunctionCode"],
    "sqs": ["sqs:SendMessage"],
    "sns": ["sns:Publish"],
}
DELETE = {
    "s3": ["s3:DeleteObject", "s3:DeleteBucket"],
    "dynamodb": ["dynamodb:DeleteItem", "dynamodb:DeleteTable"],
    "ec2": ["ec2:TerminateInstances"],
    "lambda": ["lambda:DeleteFunction"],
    "rds": ["rds:DeleteDBInstance"],
    "sqs": ["sqs:DeleteMessage", "sqs:PurgeQueue"],
}
SERVICES = ["s3", "dynamodb", "ec2", "logs", "lambda", "sqs", "rds", "cloudwatch"]
IAM_READ = ["iam:GetUser", "iam:GetRole", "iam:ListRoles", "iam:ListUsers", "iam:GetPolicy"]
IAM_ESC = sorted(a for a in PRIV_ESC_ACTIONS if a.startswith("iam:"))
SENS = ["kms:Decrypt", "secretsmanager:GetSecretValue", "ssm:GetParameter", "cloudtrail:StopLogging"]

NAMES = ["company-documents", "app-logs", "finance-reports", "hr-data", "ml-datasets",
         "backups-prod", "web-assets", "audit-archive", "customer-uploads", "dev-sandbox"]
ACCT = "123456789012"
REGION = "us-east-1"
CONDITIONS = [
    {"Bool": {"aws:MultiFactorAuthPresent": "true"}},
    {"IpAddress": {"aws:SourceIp": "203.0.113.0/24"}},
    {"StringEquals": {"aws:RequestedRegion": REGION}},
    {"StringEquals": {"aws:PrincipalTag/team": "platform"}},
]


def arn(service, name, rng):
    if service == "s3":
        return rng.choice([f"arn:aws:s3:::{name}/*", f"arn:aws:s3:::{name}", f"arn:aws:s3:::{name}/data/*"])
    if service == "dynamodb":
        return f"arn:aws:dynamodb:{REGION}:{ACCT}:table/{name}"
    if service == "sqs":
        return f"arn:aws:sqs:{REGION}:{ACCT}:{name}"
    if service == "sns":
        return f"arn:aws:sns:{REGION}:{ACCT}:{name}"
    if service == "lambda":
        return f"arn:aws:lambda:{REGION}:{ACCT}:function:{name}"
    if service == "logs":
        return f"arn:aws:logs:{REGION}:{ACCT}:log-group:/{name}:*"
    if service == "rds":
        return f"arn:aws:rds:{REGION}:{ACCT}:db:{name}"
    if service == "ec2":
        return f"arn:aws:ec2:{REGION}:{ACCT}:instance/i-{rng.randint(10**7, 10**8-1):08x}"
    if service == "kms":
        return f"arn:aws:kms:{REGION}:{ACCT}:key/{rng.randint(10**7, 10**8-1):08x}"
    if service == "secretsmanager":
        return f"arn:aws:secretsmanager:{REGION}:{ACCT}:secret:{name}-AbCdEf"
    return f"arn:aws:{service}:{REGION}:{ACCT}:{name}"


def role_arn(name):
    return f"arn:aws:iam::{ACCT}:role/{name}"


def pick(rng, pool, k_lo, k_hi):
    k = min(rng.randint(k_lo, k_hi), len(pool))
    return rng.sample(pool, k)


def stmt(actions, resource, cond=None, sid=None):
    s = {"Effect": "Allow", "Action": actions if len(actions) > 1 else actions[0], "Resource": resource}
    if sid:
        s["Sid"] = sid
    if cond:
        s["Condition"] = cond
    return s


def policy(*stmts):
    return {"Version": "2012-10-17", "Statement": list(stmts)}


# --------------------------------------------------------------------------
# Archetypes: each returns a policy given an rng. A base = (archetype, seed).
# Variations re-run the archetype with a nearby rng + small perturbations.
# --------------------------------------------------------------------------
def a_read_specific(rng):
    svc = rng.choice(list(READ))
    return policy(stmt(pick(rng, READ[svc], 1, 2), arn(svc, rng.choice(NAMES), rng)))


def a_rw_specific(rng):
    svc = rng.choice([s for s in READ if s in WRITE])
    acts = pick(rng, READ[svc], 1, 2) + pick(rng, WRITE[svc], 1, 2)
    return policy(stmt(acts, arn(svc, rng.choice(NAMES), rng)))


def a_rwd_specific(rng):
    svc = rng.choice([s for s in READ if s in WRITE and s in DELETE])
    acts = pick(rng, READ[svc], 1, 2) + pick(rng, WRITE[svc], 1, 2) + pick(rng, DELETE[svc], 1, 2)
    return policy(stmt(acts, arn(svc, rng.choice(NAMES), rng), cond=rng.choice([None, rng.choice(CONDITIONS)])))


def a_read_star_resource(rng):
    svc = rng.choice(list(READ))
    return policy(stmt(pick(rng, READ[svc], 1, 3), "*"))


def a_write_star_resource(rng):
    svc = rng.choice(list(WRITE))
    return policy(stmt(pick(rng, READ.get(svc, []), 0, 1) + pick(rng, WRITE[svc], 1, 2), "*",
                       cond=rng.choice([None, rng.choice(CONDITIONS)])))


def a_multi_service(rng):
    svcs = rng.sample([s for s in READ if s in WRITE], rng.randint(3, 4))
    stmts = []
    for svc in svcs:
        acts = pick(rng, READ[svc], 1, 2) + (pick(rng, WRITE[svc], 0, 1))
        stmts.append(stmt(acts, arn(svc, rng.choice(NAMES), rng)))
    return policy(*stmts)


def a_multi_service_delete(rng):
    svcs = rng.sample([s for s in DELETE], rng.randint(3, 4))
    return policy(*[stmt(pick(rng, READ.get(s, []), 0, 1) + pick(rng, DELETE[s], 1, 1), arn(s, rng.choice(NAMES), rng))
                    for s in svcs])


def a_service_wild_specific(rng):
    svc = rng.choice(list(READ))
    return policy(stmt([f"{svc}:*"], arn(svc, rng.choice(NAMES), rng),
                       cond=rng.choice([None, rng.choice(CONDITIONS)])))


def a_service_wild_star(rng):
    svc = rng.choice(list(READ) + ["sns"])
    return policy(stmt([f"{svc}:*"], "*"))


def a_iam_read(rng):
    return policy(stmt(pick(rng, IAM_READ, 1, 3), "*"))


def a_iam_esc_star(rng):
    return policy(stmt(pick(rng, IAM_ESC, 1, 4), "*"))


def a_iam_esc_scoped(rng):
    return policy(stmt(pick(rng, IAM_ESC, 1, 2), role_arn(rng.choice(["app-role", "etl-role", "ci-runner"])),
                       cond=rng.choice(CONDITIONS)))


def a_iam_wild(rng):
    return policy(stmt(["iam:*"], rng.choice(["*", f"arn:aws:iam::{ACCT}:role/*"])))


def a_sts_scoped(rng):
    return policy(stmt(["sts:AssumeRole"], role_arn(rng.choice(["readonly", "auditor", "deploy-role"])),
                       cond=rng.choice([None, rng.choice(CONDITIONS)])))


def a_sts_star(rng):
    return policy(stmt(["sts:AssumeRole"], "*"))


def a_sensitive_scoped(rng):
    a = rng.choice(SENS)
    svc = a.split(":")[0]
    return policy(stmt([a], arn(svc, rng.choice(NAMES), rng), cond=rng.choice([None, rng.choice(CONDITIONS)])))


def a_sensitive_star(rng):
    return policy(stmt(pick(rng, SENS, 1, 2), "*"))


def a_admin_all(rng):
    return policy(stmt(["*"], "*"))


def a_admin_scoped(rng):
    svc = rng.choice(list(READ))
    return policy(stmt(["*"], arn(svc, rng.choice(NAMES), rng), cond=rng.choice([None, rng.choice(CONDITIONS)])))


def a_notaction(rng):
    s = {"Effect": "Allow", "NotAction": rng.choice([["iam:*"], ["organizations:*"], ["account:*"]]), "Resource": "*"}
    return policy(s)


def a_mixed_read_plus_esc(rng):
    svc = rng.choice(list(READ))
    return policy(stmt(pick(rng, READ[svc], 1, 2), arn(svc, rng.choice(NAMES), rng)),
                  stmt(pick(rng, IAM_ESC, 1, 3), "*"))


def a_deny_only(rng):
    svc = rng.choice(list(READ))
    return policy({"Effect": "Deny", "Action": pick(rng, DELETE.get(svc, ["s3:DeleteObject"]), 1, 1),
                   "Resource": arn(svc, rng.choice(NAMES), rng)})


def a_mixed_scoped_plus_wild(rng):
    svc = rng.choice(list(READ))
    return policy(stmt(pick(rng, READ[svc], 1, 2), arn(svc, rng.choice(NAMES), rng)),
                  stmt([f"{rng.choice(list(READ))}:*"], "*"))


ARCHETYPES = [
    (a_read_specific, 14), (a_rw_specific, 10), (a_rwd_specific, 10), (a_read_star_resource, 8),
    (a_write_star_resource, 6), (a_multi_service, 7), (a_multi_service_delete, 5),
    (a_service_wild_specific, 7), (a_service_wild_star, 6), (a_iam_read, 4),
    (a_iam_esc_star, 6), (a_iam_esc_scoped, 5), (a_iam_wild, 4), (a_sts_scoped, 4),
    (a_sts_star, 3), (a_sensitive_scoped, 5), (a_sensitive_star, 4), (a_admin_all, 3),
    (a_admin_scoped, 3), (a_notaction, 2), (a_mixed_read_plus_esc, 4), (a_deny_only, 2),
    (a_mixed_scoped_plus_wild, 3),
]


def perturb(pol, rng):
    """Small, label-preserving-ish edits so variations differ but stay related."""
    pol = copy.deepcopy(pol)
    r = rng.random()
    stmts = pol["Statement"]
    if r < 0.2:
        rng.shuffle(stmts)
    elif r < 0.4:
        for i, s in enumerate(stmts):
            s["Sid"] = f"Stmt{rng.randint(100, 999)}"
    elif r < 0.55 and stmts[0]["Effect"] == "Allow":
        stmts[0]["Condition"] = rng.choice(CONDITIONS)       # adds a condition
    elif r < 0.65:
        for s in stmts:
            s.pop("Condition", None)                          # removes conditions
    elif r < 0.75:
        stmts.append({"Effect": "Deny", "Action": "s3:DeleteBucket",
                      "Resource": f"arn:aws:s3:::{rng.choice(NAMES)}"})
    # else: unchanged structure (names/actions already resampled)
    return pol


def build():
    rng = random.Random(SEED)
    weights = [w for _, w in ARCHETYPES]
    rows = []
    for base_id in range(N_BASES):
        fn, _ = rng.choices(ARCHETYPES, weights=weights, k=1)[0]
        base_seed = rng.randint(0, 10**9)
        for v in range(VARIATIONS):
            # variation 0 = the base; others resample details from the same archetype
            vr = random.Random(base_seed + v * 7919)
            pol = fn(vr)
            if v > 0:
                pol = perturb(pol, vr)
            feats = extract_features(pol)
            rows.append({
                "policy_id": f"policy_{len(rows) + 1:04d}",
                "group": f"base_{base_id:03d}",
                "archetype": fn.__name__,
                **feats,
                "rubric_score": round(rubric_score(pol), 2),
                "label": rubric_label(pol),
                "policy_json": json.dumps(pol),
            })
    df = pd.DataFrame(rows)
    # drop exact duplicates (same JSON) to avoid trivially repeated rows
    df = df.drop_duplicates(subset="policy_json").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = build()
    df.to_csv("policies.csv", index=False)
    print(f"{len(df)} policies, {df['group'].nunique()} base groups")
    print(df["label"].value_counts().to_string())
    print()
    print(pd.crosstab(df["archetype"], df["label"]).to_string())
