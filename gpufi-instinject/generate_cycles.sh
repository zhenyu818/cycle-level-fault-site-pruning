#!/bin/bash

# Check number of arguments
if [ $# -ne 2 ]; then
    echo "=== INFO: Usage: $0 <m> <n>"
    echo "=== INFO:   m: maximum value (range 0 to m)"
    echo "=== INFO:   n: number of values to generate"
    exit 1
fi

m=$1
n=$2

# Check if arguments are valid integers
if ! [[ "$m" =~ ^[0-9]+$ ]] || [ "$m" -lt 0 ]; then
    echo "=== INFO: Error: m must be a non-negative integer"
    exit 1
fi

if ! [[ "$n" =~ ^[0-9]+$ ]] || [ "$n" -le 0 ]; then
    echo "=== INFO: Error: n must be a positive integer"
    exit 1
fi

# Check if cycles.txt exists
if [ -f "cycles.txt" ]; then
    echo "=== INFO: Found cycles.txt, clearing file..."
    > cycles.txt
else
    echo "=== INFO: cycles.txt not found, creating file..."
    touch cycles.txt
fi

# Generate n uniformly distributed values between 0 and m
echo "=== INFO: Generating $n uniformly distributed values between 0 and $m..."

# Use awk to generate evenly spaced values
awk -v m="$m" -v n="$n" 'BEGIN {
    if (n == 1) {
        print int(m/2)
    } else {
        for (i = 0; i < n; i++) {
            value = int((i * m) / (n - 1))
            print value
        }
    }
}' > cycles.txt

echo "=== INFO: Done! $n values have been generated in cycles.txt"
