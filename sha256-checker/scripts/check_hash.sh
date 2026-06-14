#!/bin/bash
# SHA256 Checker — computes the SHA256 hash of an image file (.png, .jpg)
# Usage: ./check_hash.sh <file_path>

FILE="$1"

if [ -z "$FILE" ]; then
  echo '{"error": "No file path provided"}'
  exit 1
fi

# Check if file exists at all (including broken symlinks)
if [ ! -e "$FILE" ] && [ ! -L "$FILE" ]; then
  echo "{\"error\": \"File not found: $FILE\"}"
  exit 1
fi

# Check if it's a directory
if [ -d "$FILE" ]; then
  echo "{\"error\": \"'$FILE' is a directory, not a regular file\"}"
  exit 1
fi

# Check if it's a regular file (or a symlink pointing to one)
if [ ! -f "$FILE" ]; then
  echo "{\"error\": \"'$FILE' is not a regular file (may be a broken symlink or special file)\"}"
  exit 1
fi

# Check file extension — only .png, .jpg, .jpeg allowed
EXT="${FILE##*.}"
EXT_LOWER=$(echo "$EXT" | tr '[:upper:]' '[:lower:]')
if [ "$EXT_LOWER" != "png" ] && [ "$EXT_LOWER" != "jpg" ] && [ "$EXT_LOWER" != "jpeg" ]; then
  echo "{\"error\": \"Unsupported file type: .$EXT. Only .png, .jpg, .jpeg files are supported\"}"
  exit 1
fi

# Validate file header matches extension
HEADER_HEX=$(xxd -l 4 -p "$FILE" 2>/dev/null || od -A n -t x1 -N 4 "$FILE" 2>/dev/null | tr -d ' \n')
if [ "$EXT_LOWER" = "png" ]; then
  # PNG magic: 89 50 4E 47
  if [ "$HEADER_HEX" != "89504e47" ] && [ "$HEADER_HEX" != "89504E47" ]; then
    echo "{\"error\": \"File header does not match PNG format (expected 89 50 4E 47, got: $HEADER_HEX). File may be corrupted or misnamed.\"}"
    exit 1
  fi
elif [ "$EXT_LOWER" = "jpg" ] || [ "$EXT_LOWER" = "jpeg" ]; then
  # JPEG magic: FF D8 FF
  if [ "${HEADER_HEX:0:6}" != "ffd8ff" ] && [ "${HEADER_HEX:0:6}" != "FFD8FF" ]; then
    echo "{\"error\": \"File header does not match JPEG format (expected FF D8 FF, got: $HEADER_HEX). File may be corrupted or misnamed.\"}"
    exit 1
  fi
fi

# Check read permission
if [ ! -r "$FILE" ]; then
  echo "{\"error\": \"File exists but is not readable: $FILE (permission denied)\"}"
  exit 1
fi

# Warn on large files (>100MB)
FILESIZE=$(stat -f%z "$FILE" 2>/dev/null || stat -c%s "$FILE" 2>/dev/null)
if [ -n "$FILESIZE" ] && [ "$FILESIZE" -gt 104857600 ]; then
  FILESIZE_MB=$((FILESIZE / 1048576))
  echo "{\"warning\": \"File is ${FILESIZE_MB}MB. Computing SHA256 may take some time.\", \"continue\": true}"
fi

# Compute hash
if command -v shasum &> /dev/null; then
  HASH=$(shasum -a 256 "$FILE" | awk '{print $1}')
elif command -v sha256sum &> /dev/null; then
  HASH=$(sha256sum "$FILE" | awk '{print $1}')
else
  echo '{"error": "No SHA256 tool found (tried shasum and sha256sum)"}'
  exit 1
fi

echo "{\"file\": \"$FILE\", \"sha256\": \"$HASH\"}"
