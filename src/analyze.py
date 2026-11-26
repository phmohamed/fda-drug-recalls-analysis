"""FDA drug recalls 2012-2025: what gets recalled, why, and how serious. Plus a text classifier that predicts
Class I (most serious) from the stated reason, trained on 2012-2022 and tested on 2023-2025. From scratch, numpy only."""
import re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv("data/drug_recalls.csv")
d["year"] = d.report_date // 10000
d["reason"] = d.reason_for_recall.fillna("").str.lower()
# one "event" = same firm, same report date, same reason (a recall often covers many products/lots)
d["event"] = d.recalling_firm.fillna("") + "|" + d.report_date.astype(str) + "|" + d.reason.str[:80]
print("product records", len(d), "events", d.event.nunique())

cats = [("Sterility / contamination", r"steril|microb|contamin|bacteria|fung|mold|endotoxin|pyrogen|particulate|particle|glass|foreign"),
        ("Nitrosamine / impurity", r"nitros|nmba|ndma|ndea|impurit|degradation"),
        ("Potency (sub- or superpotent)", r"subpotent|sub-potent|superpotent|super-potent|potency|assay|out of spec.*content"),
        ("Dissolution / stability / OOS", r"dissolution|stability|out of specification|oos|specification|expir"),
        ("Labeling / packaging mix-up", r"label|barcode|packag|mix-up|mixup|wrong|incorrect|carton|insert|undeclared"),
        ("CGMP deviations", r"cgmp|good manufacturing|manufactur"),
        ("Unapproved / undeclared ingredient", r"unapproved|undeclared|not approved|adulterat|sildenafil|tadalafil")]
def cat(t):
    for n, p in cats:
        if re.search(p, t): return n
    return "Other"
d["category"] = d.reason.map(cat)
ev = d.drop_duplicates("event")

by_year = pd.DataFrame({"records": d.groupby("year").size(), "events": ev.groupby("year").size(),
                        "class_I_pct_of_events": (ev.groupby("year").classification.apply(lambda s: (s == "Class I").mean() * 100)).round(1)})
by_year.to_csv("data/recalls_by_year.csv"); print(by_year)
catt = pd.DataFrame({"events": ev.category.value_counts(), "class_I_pct": ev.groupby("category").classification.apply(lambda s: (s == "Class I").mean() * 100).round(1)})
catt["share_pct"] = (catt.events / catt.events.sum() * 100).round(1); catt.to_csv("data/reason_categories.csv"); print(catt)
top = ev.recalling_firm.value_counts().head(15).rename("events"); top.to_csv("data/top_firms_by_events.csv")
cat_year = pd.crosstab(ev.year, ev.category); cat_year.to_csv("data/reason_category_by_year.csv")

# --- classifier: Class I vs II/III from reason text ---
STOP = set("the of and in to a with for was were is are as by on at from that this these be or an which not than".split())
tok = lambda t: [w for w in re.findall(r"[a-z][a-z0-9\-]{2,}", t) if w not in STOP]
e = ev[ev.reason.str.len() > 5].reset_index(drop=True)
docs = [tok(t) for t in e.reason]; dfc = {}
for t in docs:
    for w in set(t): dfc[w] = dfc.get(w, 0) + 1
vocab = [w for w, c in sorted(dfc.items(), key=lambda x: -x[1]) if c >= 5][:3000]; ix = {w: i for i, w in enumerate(vocab)}
X = np.zeros((len(docs), len(vocab)))
for i, t in enumerate(docs):
    for w in set(t):
        if w in ix: X[i, ix[w]] = 1
y = (e.classification == "Class I").values.astype(float)
tr = (e.year <= 2022).values; te = ~tr
def fit(X, y, lam=1.0, iters=400, lr=0.5):
    w = np.zeros(X.shape[1]); b = np.log(y.mean() / (1 - y.mean()))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b))); g = p - y
        w -= lr * (X.T @ g / len(y) + lam * w / len(y)); b -= lr * g.mean()
    return w, b
def auc(y, s):
    r = pd.Series(s).rank().values; n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
w, b = fit(X[tr], y[tr]); s = X[te] @ w + b
print("train events", tr.sum(), "test events", te.sum(), "Class I share train/test", y[tr].mean().round(3), y[te].mean().round(3))
a = auc(y[te], s); print("AUC (train to 2022, test 2023+):", round(a, 3))
# base-rate by category as a simple comparison
rate = e[tr].groupby("category").classification.apply(lambda s: (s == "Class I").mean()); s2 = e[te].category.map(rate).fillna(y[tr].mean()).values
a2 = auc(y[te], s2); print("AUC using only the hand-made category:", round(a2, 3))
pd.DataFrame([{"model": "Logistic on words", "auc_2023_plus": round(a, 3)}, {"model": "Category base rate only", "auc_2023_plus": round(a2, 3)}]).to_csv("data/classifier_results.csv", index=False)
order = np.argsort(w); terms = pd.DataFrame({"class_I_terms": [vocab[i] for i in order[::-1][:15]], "class_II_III_terms": [vocab[i] for i in order[:15]]})
terms.to_csv("data/top_terms.csv", index=False); print(terms.to_string())

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(by_year.index, by_year.events, color="#4472c4"); ax.set_ylabel("Recall events (firm, date, reason)")
ax.set_title("Drug recall events by report year"); fig.tight_layout(); fig.savefig("figures/events_by_year.png", dpi=150)
fig, ax = plt.subplots(figsize=(8, 4.5)); c = catt.sort_values("events"); ax.barh(c.index, c.events, color="#4472c4")
for i, (n, r) in enumerate(zip(c.events, c.class_I_pct)): ax.text(n, i, f"  {r:.0f}% Class I", va="center", fontsize=8)
ax.set_xlabel("Recall events"); ax.set_title("Why drugs get recalled, and how often it is Class I"); fig.tight_layout(); fig.savefig("figures/reasons.png", dpi=150)
