#!/bin/bash

echo "Running simulation using config: $1"

python -m src.execution --config "$1"