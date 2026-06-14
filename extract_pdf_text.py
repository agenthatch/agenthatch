"""
PDF文本提取工具
支持原生PDF和扫描件PDF
用法: python extract_pdf_text.py <pdf文件路径> [--ocr]
"""

import sys
import argparse

def extract_native_text(pdf_path):
    """从原生PDF提取文本"""
    from pypdf import PdfReader
    
    print(f"📖 正在从原生PDF提取文本...")
    reader = PdfReader(pdf_path)
    full_text = ""
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        full_text += text + "\n"
        print(f"  第{i+1}页: {len(text)}字")
    
    return full_text


def extract_ocr_text(pdf_path, lang='chi_sim+eng'):
    """从扫描件PDF用OCR提取文本"""
    print(f"🖼️ 正在用OCR识别扫描件...")
    print(f"   语言: {lang}")
    
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("❌ 请先安装: pip install pdf2image")
        print("   系统还需安装 poppler-utils")
        sys.exit(1)
    
    try:
        import pytesseract
    except ImportError:
        print("❌ 请先安装: pip install pytesseract")
        print("   系统还需安装 tesseract-ocr")
        sys.exit(1)
    
    print(f"  正在将PDF转为图片（300 DPI）...")
    images = convert_from_path(pdf_path, dpi=300)
    print(f"  共 {len(images)} 页，开始OCR识别...")
    
    full_text = ""
    for i, img in enumerate(images):
        print(f"  正在识别第{i+1}页...")
        text = pytesseract.image_to_string(img, lang=lang)
        full_text += f"\n--- 第{i+1}页 ---\n{text}"
    
    return full_text


def main():
    parser = argparse.ArgumentParser(description='PDF文本提取工具')
    parser.add_argument('pdf_path', help='PDF文件路径')
    parser.add_argument('--ocr', action='store_true', help='强制使用OCR模式')
    parser.add_argument('--lang', default='chi_sim+eng', help='OCR语言（默认: chi_sim+eng）')
    parser.add_argument('--output', '-o', default=None, help='输出文件（默认: 输入文件名.txt）')
    
    args = parser.parse_args()
    
    # 自动判断是否需要用OCR
    use_ocr = args.ocr
    
    if not use_ocr:
        # 先用pypdf试试
        from pypdf import PdfReader
        reader = PdfReader(args.pdf_path)
        total_text = ""
        for page in reader.pages:
            total_text += (page.extract_text() or "")
        
        if len(total_text.strip()) < 100:
            print(f"⚠️ 原生文本仅有{len(total_text.strip())}字，自动切换为OCR模式")
            use_ocr = True
        else:
            print(f"✅ 检测到原生PDF，文本量: {len(total_text.strip())}字")
    
    # 提取文本
    if use_ocr:
        text = extract_ocr_text(args.pdf_path, args.lang)
    else:
        text = extract_native_text(args.pdf_path)
    
    # 保存结果
    output_path = args.output or args.pdf_path.replace('.pdf', '.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"\n{'='*50}")
    print(f"✅ 完成！共提取 {len(text)} 字")
    print(f"📁 已保存到: {output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
