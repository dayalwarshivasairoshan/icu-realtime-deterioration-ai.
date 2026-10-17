"""Run the simulation, print the results and write dashboard.html."""
import json
from pathlib import Path
from src.pipeline import run

summary, patients, alerts = run()
print("=== Results (synthetic data) ===")
for k, v in summary.items():
    print(f"{k:>18}: {v}")
red = 100 * (1 - summary["ai_alerts"] / max(summary["naive_alarms"], 1))
print(f"{'alert reduction':>18}: {red:.0f}% fewer interruptions than a fixed-threshold monitor")

data = json.dumps({"summary": summary, "patients": patients, "alerts": alerts})
html = (Path(__file__).parent / "dashboard_template.html").read_text().replace("__DATA__", data)
Path("dashboard.html").write_text(html)
print("\nWrote dashboard.html - open it in any browser.")
