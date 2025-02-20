#!/usr/bin/env bash

# Move into the directory where your images are located
# cd /path/to/images

shopt -s nullglob  # Ensure the for loop ignores if no *.jpg files

for file in *.jpg; do
  # Extract the exposure time (e.g., '1/4000' or '1/1000' or just '1')
  exposure_str=$(exiftool -ExposureTime -b "$file")

  # Convert fraction to decimal if needed.
  # e.g., 1/4000 -> 0.00025, 1/750 -> 0.00133, etc.
  if [[ "$exposure_str" =~ ^([0-9]+)/([0-9]+)$ ]]; then
    # It's a fraction 'numerator/denominator'
    numerator="${BASH_REMATCH[1]}"
    denominator="${BASH_REMATCH[2]}"
    # Use awk for floating-point arithmetic
    exposure_dec=$(awk -v n="$numerator" -v d="$denominator" 'BEGIN { printf "%.6f", n/d }')
  else
    # It's already a plain number (like '1')
    # Possibly convert to float with some decimal places
    exposure_dec=$(awk -v e="$exposure_str" 'BEGIN { printf "%.6f", e }')
  fi

  # Construct the new filename: uv_<exposure_seconds>s.jpg
  # Example: uv_0.00025s.jpg
  random_number=$(shuf -i 1-1000 -n 1)
  new_name="uv_${exposure_dec}s_${random_number}.png"

  echo "Renaming '$file' -> '$new_name'"
  mv -- "$file" "$new_name"
done

echo "Done renaming based on exposure times."
