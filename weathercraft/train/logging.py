from __future__ import annotations

import csv
from pathlib import Path


class CsvLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames: list[str] | None = None

    def write(self, row: dict[str, float | int | str]) -> None:
        if self.fieldnames is None:
            self.fieldnames = list(row)
            if self.path.exists() and self.path.stat().st_size:
                with self.path.open("r", newline="", encoding="utf-8") as handle:
                    self.fieldnames = next(csv.reader(handle))
        unknown = set(row) - set(self.fieldnames)
        if unknown:
            raise ValueError(f"new CSV fields appeared after initialization: {sorted(unknown)}")
        first_write = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if first_write:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in self.fieldnames})
