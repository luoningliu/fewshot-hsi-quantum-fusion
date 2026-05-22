from __future__ import annotations

from pathlib import Path

import numpy as np


def main() -> None:
    rng = np.random.default_rng(42)
    cities = np.array(["Berlin", "Munich", "Cologne"])
    lcz_labels = np.array([1, 4, 8, 11, 17])
    samples_per_city_label = 16

    x_rows = []
    y_rows = []
    city_rows = []
    for city in cities:
        for label in lcz_labels:
            for _ in range(samples_per_city_label):
                base = float(label) / 20.0
                patch = rng.normal(loc=base, scale=0.05, size=(16, 16, 4)).astype(np.float32)
                x_rows.append(patch)
                y_rows.append(label)
                city_rows.append(city)

    out = Path("data/lcz42/raw/synthetic_lcz42_sentinel2.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        x=np.stack(x_rows),
        y=np.asarray(y_rows),
        city=np.asarray(city_rows),
        bands=np.asarray(["B2", "B3", "B4", "B8"]),
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

