from __future__ import annotations

import json

from ucimlrepo import fetch_ucirepo

from common import DATA_DIR, ensure_dirs


def main() -> None:
    ensure_dirs()
    dataset = fetch_ucirepo(id=468)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    target_name = targets.columns[0]
    features.assign(**{target_name: targets.iloc[:, 0]}).to_csv(DATA_DIR / "online_shoppers_purchasing_intention.csv", index=False)
    (DATA_DIR / "metadata.json").write_text(json.dumps(dataset.metadata, indent=2, default=str) + "\n")
    dataset.variables.to_csv(DATA_DIR / "variables.csv", index=False)
    print(f"Saved {len(features):,} rows and {len(features.columns)} features to {DATA_DIR}")


if __name__ == "__main__":
    main()