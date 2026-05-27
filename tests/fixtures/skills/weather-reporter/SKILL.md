---
name: weather-reporter
description: Query current weather and forecast data from OpenWeatherMap API. Use when the user asks about weather, temperature, forecast, humidity, wind, or precipitation for any location.
compatibility: Requires OPENWEATHER_API_KEY environment variable and internet access
metadata:
  author: agenthatch-labs
  version: "1.0.0"
allowed-tools: Bash(curl:*) Bash(jq:*) Read
---

# Weather Reporter

Query weather data from OpenWeatherMap API.

## Quick Start

```bash
curl -s "https://api.openweathermap.org/data/2.5/weather?q=${CITY}&appid=${OPENWEATHER_API_KEY}&units=metric" | jq .
```

## Workflow

1. Extract city name from user query
2. Call the API with `scripts/query_weather.sh <city>`
3. Parse JSON response and format as human-readable text
4. For forecasts, use `scripts/forecast_weather.sh <city> <days>`

## Output Format

```
Weather in {city}, {country}:
  Temperature: {temp}°C (feels like {feels_like}°C)
  Condition: {description}
  Humidity: {humidity}%
  Wind: {wind_speed} m/s
```

## Gotchas

- City names with spaces must be URL-encoded (e.g., "New York" → "New%20York")
- The free API tier limits to 60 calls/min
- Always specify `units=metric` unless user explicitly requests imperial