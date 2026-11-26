"""Download FDA drug recall (enforcement report) records from openFDA, 2012 to 2025."""
import requests, pandas as pd, time
rows = []
for yr in range(2012, 2026):
    for skip in range(0, 5000, 1000):
        r = requests.get("https://api.fda.gov/drug/enforcement.json", params={
            "search": f"report_date:[{yr}0101 TO {yr}1231]", "limit": 1000, "skip": skip}, timeout=90)
        if r.status_code != 200: break
        res = r.json()["results"]
        for x in res:
            rows.append({k: x.get(k) for k in ["recall_number", "report_date", "classification", "recalling_firm", "reason_for_recall",
                                             "product_description", "voluntary_mandated", "status", "country", "product_type", "recall_initiation_date", "distribution_pattern"]})
        if len(res) < 1000: break
        time.sleep(0.3)
    print(yr, len(rows))
df = pd.DataFrame(rows).drop_duplicates("recall_number"); df.to_csv("data/drug_recalls.csv", index=False); print(len(df))
