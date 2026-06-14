"""
PDF类型判断与文本提取工具
用法: python check_pdf_type.py <pdf文件路径>
"""

import sys
from pypdf import PdfReader

def analyze_pdf(filepath):
    """分析PDF文件，判断是扫描件还是原生PDF"""
    print(f"\n{'='*50}")
    print(f"📄 正在分析: {filepath}")
    print(f"{'='*50}\n")
    
    try:
        reader = PdfReader(filepath)
    except Exception as e:
        print(f"❌ 无法打开文件: {e}")
        return None
    
    total_pages = len(reader.pages)
    total_text_len = 0
    total_images = 0
    page_details = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text_len = len(text.strip())
        img_count = len(page.images)
        
        total_text_len += text_len
        total_images += img_count
        
        page_details.append({
            "page_num": i + 1,
            "text_len": text_len,
            "img_count": img_count,
            "text_preview": text[:80].replace("\n", " ") if text else "(空)"
        })
        
        # 打印每页信息
        icon = "🖼️" if img_count > 0 else "📝"
        print(f"  第{i+1:2d}页 | 文本:{text_len:5d}字 | 图片:{img_count}张 {icon}")
    
    # 打印摘要
    print(f"\n{'─'*50}")
    print(f"📊 分析摘要")
    print(f"{'─'*50}")
    print(f"  总页数:     {total_pages}")
    print(f"  总文本长度: {total_text_len} 字")
    print(f"  总图片数:   {total_images}")
    
    # ========== 判断逻辑 ==========
    print(f"\n{'─'*50}")
    print(f"🔍 判断结果")
    print(f"{'─'*50}")
    
    # 判断依据
    has_text = total_text_len > 100
    image_density = total_images / total_pages if total_pages > 0 else 0
    
    if has_text and image_density < 0.5:
        print(f"\n✅ 结论：这是【原生PDF】")
        print(f"   理由：包含 {total_text_len} 字可选文字，图片较少")
        print(f"\n📌 推荐提取方案：")
        print(f"   工具: pypdf / pdfplumber")
        print(f"   命令: python extract_text.py {filepath}")
        
        # 显示文本预览
        print(f"\n📖 文本预览（前300字）：")
        print(f"{'─'*40}")
        full_text = ""
        for page in reader.pages:
            full_text += (page.extract_text() or "")
        print(full_text[:300])
        if len(full_text) > 300:
            print("...（更多内容省略）")
        
        return "native"
        
    elif not has_text and image_density >= 0.5:
        print(f"\n🖼️ 结论：这是【扫描件PDF】")
        print(f"   理由：文本极少（{total_text_len}字），页面包含 {total_images} 张扫描图片")
        print(f"\n📌 推荐提取方案：")
        print(f"   工具: OCR（光学字符识别）")
        print(f"   方案一（pytesseract）:")
        print(f"     pip install pdf2image pytesseract")
        print(f"     python -c \"from pdf2image import convert_from_path;")
        print(f"     import pytesseract;")
        print(f"     images = convert_from_path('{filepath}', dpi=300);")
        print(f"     for i,img in enumerate(images): print(pytesseract.image_to_string(img, lang='chi_sim+eng'))\"")
        print(f"   方案二（easyocr - 更简单）:")
        print(f"     pip install pdf2image easyocr")
        print(f"     python -c \"import easyocr; from pdf2image import convert_from_path;")
        print(f"     reader = easyocr.Reader(['ch_sim','en']);")
        print(f"     for img in convert_from_path('{filepath}', dpi=300):")
        print(f"       print(' '.join([t[1] for t in reader.readtext(img)]))\"")
        
        return "scanned"
        
    else:
        print(f"\n⚠️ 结论：这是【混合型PDF】")
        print(f"   理由：包含部分文本（{total_text_len}字）和 {total_images} 张图片")
        print(f"\n📌 推荐提取方案：")
        print(f"   先用 pypdf 提取可选文字，再用 OCR 补充图片中的文字")
        return "mixed"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_pdf_type.py <pdf文件路径>")
        print("示例: python check_pdf_type.py scan_report.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    result = analyze_pdf(pdf_path)
    
    if result:
        print(f"\n{'='*50}")
        print(f"✅ 分析完成！PDF类型: {result}")
        print(f"{'='*50}\n")
