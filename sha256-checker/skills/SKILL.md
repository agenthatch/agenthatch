---
name: Image SHA256 Checker
description: >
  Calculate the SHA256 hash of .png and .jpg image files.
  Trigger when the user mentions 'hash', 'hasher', '校验', 'checksum',
  or asks to verify image integrity or compare image hashes.
version: "2.1.0"
allowed_tools:
  - bash_runtime
category: utility
tags:
  - hash
  - sha256
  - checksum
  - image
  - file-integrity
trigger:
  keywords:
    - hash
    - hasher
    - 校验
    - checksum
---

# Image SHA256 Checker

Computes the SHA256 hash of **.png** and **.jpg** (including .jpeg) image files.

## What This Skill Does

When a user provides an image file path and mentions hash/checksum-related keywords, this skill:

1. Validates that the file exists and is a regular file
2. Checks that the file extension is `.png`, `.jpg`, or `.jpeg`
3. (Optional) Verifies the file header matches the expected image format
4. Verifies the file is readable (has proper permissions)
5. Warns the user if the file is large (>100MB) before computing
6. Computes the SHA256 hash using `shasum -a 256` (macOS) or `sha256sum` (Linux)
7. Returns the hash in a clean, readable format

## Instructions

1. **Extract the file path** from the user's request. If no path is given, ask for it.
2. **Verify the file exists** — if not, inform the user and stop.
3. **Check the file extension** — only `.png`, `.jpg`, `.jpeg` are allowed. If the file is not an image of these types, inform the user and stop.
4. **(Optional) Validate file header** — for `.png` files, verify the file starts with the PNG magic bytes (`\x89PNG`); for `.jpg` files, verify it starts with `\xFF\xD8\xFF`. If the header doesn't match, warn the user that the file may be corrupted or misnamed.
5. **Check read permission** — if the file exists but can't be read, inform the user.
6. **Warn on large files** — if the file size exceeds 100MB, inform the user that hash computation may take some time, and ask if they want to proceed.
7. **Run the hash command** using the provided script:

   ```bash
   bash scripts/check_hash.sh "/path/to/file"
   ```

8. **Format the output** clearly, showing:
   - The filename
   - The full SHA256 hash (64-character hex string)
   - If the user provides a known hash to compare against, indicate whether they match

## Output Template

```
File: {filename}
SHA256: {hex_hash}
```

If the user is comparing against a known hash:

```
File: {filename}
SHA256: {hex_hash}
Match: ✅ YES / ❌ NO
```

## Error Handling

The script returns structured JSON error messages for each failure case. When the skill encounters an error, present it to the user in plain language:

| Scenario | Error Message | User-Facing Response |
|---|---|---|
| No path given | `"No file path provided"` | "请提供要计算哈希值的图片文件路径。" |
| File not found | `"File not found: ..."` | "文件不存在，请检查路径是否正确。" |
| Path is a directory | `"... is a directory ..."` | "这是一个目录，不是文件。请提供图片文件路径。" |
| Not a regular file | `"... is not a regular file ..."` | "该路径不是普通文件（可能是损坏的符号链接或特殊文件）。" |
| Unsupported extension | `"Unsupported file type: ..."` | "不支持该文件类型。仅支持 .png、.jpg、.jpeg 格式的图片文件。" |
| Header mismatch | `"File header does not match ..."` | "文件头与扩展名不匹配，文件可能已损坏或扩展名有误。" |
| Permission denied | `"... not readable ..."` | "文件存在但无读取权限，请检查文件权限设置。" |
| No hash tool | `"No SHA256 tool found ..."` | "系统未安装 SHA256 计算工具（需要 shasum 或 sha256sum）。" |

## Constraints

- Only operate on files the agent already has access to — do **not** download files from the internet
- **Only supports .png, .jpg, .jpeg files** — reject any other file type with a clear message
- If the file path contains spaces, ensure proper quoting in the shell command
- Do **not** modify the file in any way
- If the file is a directory, inform the user that this skill only works on image files
- If the file does not exist, report that clearly and do not proceed
- For files larger than 100MB, ask the user for confirmation before proceeding
