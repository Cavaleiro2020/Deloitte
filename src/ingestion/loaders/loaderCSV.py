from __future__ import annotations

import csv
from pathlib import Path

from src.ingestion.loaders.loaderBase import LoaderBase


class LoaderCSV(LoaderBase):
    """Loader for sustainable product recommendation CSV files."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._records: list[dict] | None = None

    def _split_list_field(self, value: str | None) -> list[str]:
        if not value:
            return []
        return [item.strip() for item in value.split(";") if item.strip()]

    def _parse_float(self, value: str | None, default: float = 0.0) -> float:
        try:
            return float((value or "").replace("$", "").strip())
        except ValueError:
            return default

    def _parse_int(self, value: str | None, default: int = 0) -> int:
        try:
            return int(float((value or "").strip()))
        except ValueError:
            return default

    def _normalize_row(self, row: dict[str, str]) -> dict:
        return {
            "product_id": self._parse_int(row.get("Product ID")),
            "name": (row.get("Product Name") or "").strip(),
            "description": (row.get("Description") or "").strip(),
            "category": (row.get("Category") or "").strip(),
            "colors": self._split_list_field(row.get("Colors")),
            "review_score": self._parse_float(row.get("Review Score")),
            "countries": self._split_list_field(row.get("Countries")),
            "price": self._parse_float(row.get("Price ($)")),
        }

    def load_products(self) -> list[dict]:
        if self._records is not None:
            return self._records

        records: list[dict] = []
        with open(self.filepath, "r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                normalized = self._normalize_row(row)
                if normalized["name"]:
                    records.append(normalized)

        self._records = records
        return records

    def extract_metadata(self):
        products = self.load_products()
        columns = list(products[0].keys()) if products else []

        return {
            "file_name": Path(self.filepath).name,
            "row_count": len(products),
            "columns": columns,
        }

    def extract_text(self):
        lines = []
        for product in self.load_products():
            countries = ", ".join(product["countries"]) if product["countries"] else "Unknown"
            colors = ", ".join(product["colors"]) if product["colors"] else "Unknown"
            lines.append(
                f"{product['name']} | {product['category']} | ${product['price']:.2f} | "
                f"rating {product['review_score']:.1f} | colors: {colors} | countries: {countries} | {product['description']}"
            )
        return "\n".join(lines)