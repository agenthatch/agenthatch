#!/bin/bash
# Query OpenWeatherMap API for current weather
CITY="$1"
curl -s "https://api.openweathermap.org/data/2.5/weather?q=${CITY}&appid=${OPENWEATHER_API_KEY}&units=metric" | jq .