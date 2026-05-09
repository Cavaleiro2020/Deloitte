from __future__ import annotations

import re
from pathlib import Path

from src.ingestion.loaders.loaderCSV import LoaderCSV


class ProductRecommender:
    """Structured recommender for sustainable product queries."""

    CATEGORY_SYNONYMS = {
        "Reusable Items": {"reusable", "refillable", "zero waste", "plastic free", "plastic-free"},
        "Personal Care": {"personal care", "toothbrush", "deodorant", "shampoo", "soap", "lip balm", "razor", "comb"},
        "Renewable Energy": {"solar", "energy generation", "renewable", "power bank", "charger", "lantern", "radio", "watch", "bike light"},
        "Energy Efficiency": {"energy efficient", "energy efficiency", "efficient", "thermostat", "light bulb", "bulb", "showerhead", "curtain", "heater", "power strip", "lamp", "lights"},
        "Sustainable Fashion": {"fashion", "clothing", "shirt", "t-shirt", "jeans", "socks", "sunglasses", "backpack", "wallet", "jacket", "scarf", "sneakers"},
    }

    QUERY_CUES = {
        "recommend",
        "recommendation",
        "suggest",
        "product",
        "products",
        "item",
        "items",
        "buy",
        "purchase",
        "shopping",
        "gift",
        "budget",
        "cheap",
        "affordable",
        "under",
        "below",
        "less than",
    }

    def __init__(self, csv_path: str | None = None):
        self.csv_path = csv_path or str(Path("data") / "sustainable_products.csv")
        self.loader = LoaderCSV(self.csv_path)
        self.products = self.loader.load_products()
        self.available_countries = self._build_available_countries()

    def _build_available_countries(self) -> set[str]:
        countries: set[str] = set()
        for product in self.products:
            for country in product.get("countries", []):
                countries.add(country.lower())
        return countries

    def _tokenize(self, query: str) -> set[str]:
        tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "for",
            "to",
            "of",
            "in",
            "on",
            "with",
            "me",
            "show",
            "give",
            "some",
            "please",
            "need",
            "want",
            "looking",
            "find",
        }
        return {token for token in tokens if len(token) > 2 and token not in stop_words}

    def _extract_budget_limit(self, query: str) -> float | None:
        match = re.search(
            r"(?:under|below|less than|max(?:imum)?|budget(?: of)?|up to)\s*\$?\s*(\d+(?:\.\d+)?)",
            query.lower(),
        )
        if match:
            return float(match.group(1))

        match = re.search(r"\$\s*(\d+(?:\.\d+)?)", query.lower())
        if match and any(cue in query.lower() for cue in ("under", "below", "less", "budget", "affordable", "cheap")):
            return float(match.group(1))

        return None

    def _extract_country_preferences(self, query: str) -> list[str]:
        query_lower = query.lower()
        matches = []
        for country in sorted(self.available_countries):
            if country in query_lower:
                matches.append(country)
        return matches

    def _category_from_query(self, query: str) -> str | None:
        query_lower = query.lower()
        for category, synonyms in self.CATEGORY_SYNONYMS.items():
            if category.lower() in query_lower:
                return category
            if any(synonym in query_lower for synonym in synonyms):
                return category
        return None

    def is_product_query(self, query: str) -> bool:
        query_lower = query.lower()

        if any(cue in query_lower for cue in self.QUERY_CUES):
            return True

        if self._category_from_query(query) is not None:
            return True

        if self._extract_budget_limit(query) is not None:
            return True

        return False

    def _score_product(self, product: dict, query: str) -> tuple[float, list[str]]:
        query_lower = query.lower()
        query_tokens = self._tokenize(query)
        product_text = " ".join(
            [
                product.get("name", ""),
                product.get("description", ""),
                product.get("category", ""),
                " ".join(product.get("colors", [])),
            ]
        ).lower()

        score = 0.0
        reasons: list[str] = []

        overlap = sorted(token for token in query_tokens if token in product_text)
        if overlap:
            score += len(overlap) * 1.8
            reasons.append(f"Matched keywords: {', '.join(overlap[:4])}")

        category = product.get("category", "")
        category_synonyms = self.CATEGORY_SYNONYMS.get(category, set())
        if category and (category.lower() in query_lower or any(synonym in query_lower for synonym in category_synonyms)):
            score += 6.0
            reasons.append(f"Category fit: {category}")

        review_score = float(product.get("review_score", 0.0))
        score += review_score * 1.4
        reasons.append(f"Strong review score: {review_score:.1f}/5")

        price = float(product.get("price", 0.0))
        budget_limit = self._extract_budget_limit(query)
        if budget_limit is not None:
            if price <= budget_limit:
                score += 5.0 + max(0.0, (budget_limit - price) / max(budget_limit, 1.0)) * 3.0
                reasons.append(f"Within budget: ${price:.2f} <= ${budget_limit:.2f}")
            else:
                penalty = min(5.0, ((price - budget_limit) / max(budget_limit, 1.0)) * 5.0)
                score -= penalty

        if any(word in query_lower for word in ("cheap", "affordable", "budget", "under", "below")):
            score += max(0.0, 4.0 - (price / 50.0))
            reasons.append("Pricing fits a budget-friendly request")

        preferred_countries = self._extract_country_preferences(query)
        if preferred_countries:
            product_countries = {country.lower() for country in product.get("countries", [])}
            country_matches = sorted(product_countries.intersection(preferred_countries))
            if country_matches:
                score += 3.5 + len(country_matches)
                reasons.append(f"Available in: {', '.join(country_matches[:3])}")
            else:
                score -= 1.5

        if any(word in query_lower for word in ("best", "top", "highest rated", "most rated")):
            score += review_score * 0.8

        return score, reasons

    def recommend(self, query: str, limit: int = 5) -> list[dict]:
        ranked_products = []

        for product in self.products:
            score, reasons = self._score_product(product, query)
            ranked_products.append(
                {
                    **product,
                    "score": round(score, 2),
                    "match_reasons": reasons,
                }
            )

        ranked_products.sort(key=lambda item: (item["score"], item["review_score"], -item["price"]), reverse=True)
        return ranked_products[:limit]

    def format_recommendations(self, products: list[dict], query: str) -> str:
        if not products:
            return (
                "I could not find a strong product match for that request. "
                "Try adding a category, budget, or country preference."
            )

        lines = [
            "### Sustainable product recommendations",
            f"Source: {Path(self.csv_path).name}",
            f"Query: {query}",
            "",
            "| Rank | Product | Category | Price | Rating | Available in | Why it matched |",
            "| --- | --- | --- | ---: | ---: | --- | --- |",
        ]

        for index, product in enumerate(products, start=1):
            countries = ", ".join(product.get("countries", [])) or "Not specified"
            reasons = "; ".join(product.get("match_reasons", [])) or "Relevant sustainable option"
            lines.append(
                f"| {index} | {product.get('name', '')} | {product.get('category', '')} | "
                f"${float(product.get('price', 0.0)):.2f} | {float(product.get('review_score', 0.0)):.1f}/5 | "
                f"{countries} | {reasons} |"
            )

        return "\n".join(lines)