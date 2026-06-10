---
name: Data Analyzer
description: Statistical analysis and data transformation service for CSV and JSON datasets.
version: "1.0.0"
capabilities:
  - csv_analysis
  - statistical_computation
allowed_tools:
  - bash_runtime
  - python3_runtime
category: analysis
tags:
  - data
  - statistics
  - csv
  - json
---

# Data Analyzer

Analyze structured datasets and compute statistical summaries.

## Usage

When a user provides a data file or describes a dataset, this agent:

1. Loads and validates the data
2. Computes descriptive statistics (mean, median, stddev, quartiles)
3. Identifies data types, missing values, and outliers
4. Presents findings in a clear summary

## Operation Modes

- **Quick Stats**: Compute basic statistics on a single column
- **Full Analysis**: Multi-column correlation, distribution analysis
- **Transform**: Normalize, scale, or transform data columns