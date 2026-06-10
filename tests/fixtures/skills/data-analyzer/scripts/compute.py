"""Statistical computation functions for Data Analyzer skill.

These functions are designed to be import-bound in v0.8.0.
Each function has typed parameters, a descriptive docstring with
Sphinx-style :param annotations, and a typed return value.
"""



def compute_stats(values: list[float]) -> dict:
    """Compute descriptive statistics for a list of numeric values.

    :param values: List of numeric values to analyze
    :returns: dict with mean, median, std, min, max, q1, q3, count, missing_count
    """
    n = len(values)
    if n == 0:
        return {"error": "empty input", "count": 0}

    sorted_values = sorted(values)
    mean_val = sum(values) / n
    median_idx = n // 2
    median_val = (
        sorted_values[median_idx]
        if n % 2 == 1
        else (sorted_values[median_idx - 1] + sorted_values[median_idx]) / 2
    )

    # Standard deviation
    variance = sum((x - mean_val) ** 2 for x in values) / n
    std_val = variance ** 0.5

    # Quartiles
    q1 = sorted_values[n // 4]
    q3 = sorted_values[3 * n // 4]

    return {
        "count": n,
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "std": round(std_val, 4),
        "min": round(sorted_values[0], 4),
        "max": round(sorted_values[-1], 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "range": round(sorted_values[-1] - sorted_values[0], 4),
    }


def detect_outliers(values: list[float], threshold: float = 2.0) -> list[float]:
    """Detect outliers using the z-score method.

    :param values: List of numeric values
    :param threshold: Z-score threshold for outlier detection (default 2.0)
    :returns: List of outlier values
    """
    n = len(values)
    if n < 3:
        return []

    mean_val = sum(values) / n
    std_val = (sum((x - mean_val) ** 2 for x in values) / n) ** 0.5

    if std_val == 0:
        return []

    outliers = []
    for v in values:
        z_score = abs((v - mean_val) / std_val)
        if z_score > threshold:
            outliers.append(v)

    return outliers


async def load_csv(path: str, delimiter: str = ",") -> dict:
    """Load and parse a CSV file into column-oriented data.

    :param path: Path to the CSV file
    :param delimiter: Column delimiter character (default comma)
    :returns: dict with headers list and column data
    """
    import csv
    from pathlib import Path

    csv_path = Path(path)
    if not csv_path.exists():
        return {"error": f"File not found: {path}", "headers": [], "columns": {}}

    with open(csv_path, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        headers = next(reader, [])
        rows = list(reader)

    if not headers:
        return {"error": "Empty CSV", "headers": [], "columns": {}}

    columns = {h: [row[i] if i < len(row) else "" for row in rows] for i, h in enumerate(headers)}
    return {"headers": headers, "columns": columns, "row_count": len(rows)}


# Private helper — should NOT be extracted by Phase 1.5
def _validate_numeric(values: list[float]) -> bool:
    """Internal validation helper."""
    return all(isinstance(v, (int, float)) for v in values)
