# PDF类型判断与文本提取工具

## 快速开始

### 1️⃣ 判断PDF类型

```bash
python check_pdf_type.py scan_report.pdf
```

输出示例：
```
==================================================
📄 正在分析: scan_report.pdf
==================================================

  第 1页 | 文本:    0字 | 图片:1张 🖼️
  第 2页 | 文本:    0字 | 图片:1张 🖼️
  第 3页 | 文本:    0字 | 图片:1张 🖼️

──────────────────────────────────────────────────
🔍 判断结果
──────────────────────────────────────────────────

🖼️ 结论：这是【扫描件PDF】
   理由：文本极少（0字），页面包含 3 张扫描图片

📌 推荐提取方案：
   工具: OCR（光学字符识别）
   方案一（pytesseract）: ...
```

### 2️⃣ 提取文本

**自动模式**（智能判断类型）：
```bash
python extract_pdf_text.py scan_report.pdf
```

**强制OCR模式**：
```bash
python extract_pdf_text.py scan_report.pdf --ocr
```

**指定语言**（如纯英文）：
```bash
python extract_pdf_text.py scan_report.pdf --lang eng
```

### 3️⃣ 安装依赖

```bash
# 基础
pip install pypdf pdfplumber

# OCR（扫描件需要）
pip install pdf2image pytesseract easyocr

# 系统依赖
# Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-chi-sim poppler-utils
# macOS:  brew install tesseract poppler
# Windows: 下载安装 Tesseract OCR + poppler
```

## 判断方法速查表

| 方法 | 命令/代码 | 原生PDF特征 | 扫描件特征 |
|------|-----------|-------------|-----------|
| Python检查 | `check_pdf_type.py` | 文本>100字 | 文本≈0, 图片多 |
| 命令行 | `pdftotext file.pdf - \| head` | 输出正常文字 | 输出空或乱码 |
| 手动 | 打开PDF尝试选中文字 | ✅ 可以选中复制 | ❌ 选不中 |
| 手动 | Ctrl+F搜索 | ✅ 能搜到 | ❌ 搜不到 |
