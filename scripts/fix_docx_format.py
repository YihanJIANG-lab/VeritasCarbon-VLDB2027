"""
Fix docx formatting to match SIGMOD interim-layout.docx template.
Issues fixed:
1. Margins: all sections → top=1500, right=1080, bottom=1600, left=1080, header=1080, footer=1080
2. Footnote/copyright block (P7-P12): apply correct template styles
3. Figure 3 section: move body text paragraph out of the single-column figure section
   so that only the image + caption remain in single-column
4. Ensure consistent column spacing
"""
import zipfile
import shutil
import copy
import os
from xml.etree import ElementTree as ET

# Register namespaces to preserve them
namespaces = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
    'mc': 'http://schemas.openxmlformats.org/markup-compatibility/2006',
    'wps': 'http://schemas.microsoft.com/office/word/2010/wordprocessingShape',
    'w14': 'http://schemas.microsoft.com/office/word/2010/wordml',
    'w15': 'http://schemas.microsoft.com/office/word/2012/wordml',
    'wpc': 'http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas',
    'wpg': 'http://schemas.microsoft.com/office/word/2010/wordprocessingGroup',
    'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
    'o': 'urn:schemas-microsoft-com:office:office',
    'v': 'urn:schemas-microsoft-com:vml',
    'wne': 'http://schemas.microsoft.com/office/word/2006/wordml',
    'sl': 'http://schemas.openxmlformats.org/schemaLibrary/2006/main',
}
for prefix, uri in namespaces.items():
    ET.register_namespace(prefix, uri)

ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'

INPUT_PATH = 'docs/2026-04-10-VeritasCarbon Traceable Instruction D_replaced.docx'
OUTPUT_PATH = 'docs/2026-04-10-VeritasCarbon Traceable Instruction D_formatted.docx'

# Template margins (in twips)
TEMPLATE_MARGINS = {
    'top': '1500', 'right': '1080', 'bottom': '1600',
    'left': '1080', 'header': '1080', 'footer': '1080', 'gutter': '0'
}

# Template footnote/copyright styles mapping
# P7 → FootnoteText, P8 → FootnoteText, P9 → PermissionBlock,
# P10 → VersoLRH, P11 → PermissionBlock, P12 → (doi link)
FOOTNOTE_STYLE_MAP = {
    '∗Article Title Footnote': 'FootnoteText',
    '†Author Footnote': 'FootnoteText',
    'Permission to make digital': 'PermissionBlock',
    'WOODSTOCK': 'VersoLRH',
    '© 2018 Copyright': 'PermissionBlock',
    'https://doi.org': 'PermissionBlock',
}


def get_para_text(para):
    return ''.join(t.text or '' for t in para.iter(f'{W}t'))


def fix_margins(sectPr):
    """Fix margins in a sectPr element to match template."""
    pgMar = sectPr.find(f'{W}pgMar')
    if pgMar is not None:
        for attr, val in TEMPLATE_MARGINS.items():
            pgMar.set(f'{W}{attr}', val)


def ensure_style_exists(styles_root, style_id, style_name, font='Linux Libertine', sz='18'):
    """Add a style to styles.xml if it doesn't exist."""
    existing = styles_root.find(f".//w:style[@{W}styleId='{style_id}']", ns)
    if existing is not None:
        return

    style = ET.SubElement(styles_root, f'{W}style')
    style.set(f'{W}type', 'paragraph')
    style.set(f'{W}styleId', style_id)

    name_el = ET.SubElement(style, f'{W}name')
    name_el.set(f'{W}val', style_name)

    basedOn = ET.SubElement(style, f'{W}basedOn')
    basedOn.set(f'{W}val', 'Normal')

    pPr = ET.SubElement(style, f'{W}pPr')
    spacing = ET.SubElement(pPr, f'{W}spacing')
    spacing.set(f'{W}after', '0')
    spacing.set(f'{W}line', '240')
    spacing.set(f'{W}lineRule', 'auto')

    rPr = ET.SubElement(style, f'{W}rPr')
    sz_el = ET.SubElement(rPr, f'{W}sz')
    sz_el.set(f'{W}val', sz)
    szCs = ET.SubElement(rPr, f'{W}szCs')
    szCs.set(f'{W}val', sz)
    rFonts = ET.SubElement(rPr, f'{W}rFonts')
    rFonts.set(f'{W}ascii', font)
    rFonts.set(f'{W}hAnsi', font)


def set_para_style(para, style_id):
    """Set or replace the paragraph style."""
    pPr = para.find(f'{W}pPr')
    if pPr is None:
        pPr = ET.SubElement(para, f'{W}pPr')
        # Move to beginning
        para.remove(pPr)
        para.insert(0, pPr)
    pStyle = pPr.find(f'{W}pStyle')
    if pStyle is None:
        pStyle = ET.SubElement(pPr, f'{W}pStyle')
        pPr.remove(pStyle)
        pPr.insert(0, pStyle)
    pStyle.set(f'{W}val', style_id)


def process_document():
    # Copy the file
    shutil.copy2(INPUT_PATH, OUTPUT_PATH)

    with zipfile.ZipFile(OUTPUT_PATH, 'r') as zin:
        doc_xml = zin.read('word/document.xml')
        styles_xml = zin.read('word/styles.xml')
        all_files = {name: zin.read(name) for name in zin.namelist()}

    doc_root = ET.fromstring(doc_xml)
    styles_root = ET.fromstring(styles_xml)
    body = doc_root.find(f'.//w:body', ns)

    # ===== 1. Add missing styles =====
    print("1. Adding missing template styles...")
    ensure_style_exists(styles_root, 'FootnoteText', 'FootnoteText',
                        font='Linux Libertine', sz='14')
    ensure_style_exists(styles_root, 'PermissionBlock', 'PermissionBlock',
                        font='Linux Libertine', sz='14')
    ensure_style_exists(styles_root, 'VersoLRH', 'VersoLRH',
                        font='Linux Libertine', sz='14')

    # ===== 2. Fix margins in ALL section properties =====
    print("2. Fixing margins in all sections...")
    sect_count = 0
    for sectPr in doc_root.iter(f'{W}sectPr'):
        fix_margins(sectPr)
        sect_count += 1
    print(f"   Fixed {sect_count} section properties")

    # ===== 3. Fix footnote/copyright block styles =====
    print("3. Fixing footnote/copyright block styles...")
    children = list(body)
    for para in children:
        if para.tag != f'{W}p':
            continue
        text = get_para_text(para)
        for prefix, style_id in FOOTNOTE_STYLE_MAP.items():
            if text.strip().startswith(prefix):
                set_para_style(para, style_id)
                print(f"   Applied style '{style_id}' to: {text[:60]}...")
                break

    # ===== 4. Fix Figure 3 section: move body text out of single-column section =====
    print("4. Fixing Figure 3 section layout...")
    # Find the section break that transitions to 1-col for Figure 3
    # The paragraph with body text "The main intrinsic evaluation..." is currently
    # in the 1-col section with the figure - it should be in the preceding 2-col section.
    #
    # Strategy: Find section break #4 (before Figure 3), and move it so that
    # "The main intrinsic evaluation..." paragraph stays in the 2-col section.
    
    children = list(body)
    figure3_text_para = None
    figure3_break_para = None  # The paragraph containing the section break before Figure 3
    
    for i, child in enumerate(children):
        if child.tag != f'{W}p':
            continue
        text = get_para_text(child)
        
        # Find section break that precedes Figure 3
        pPr = child.find(f'{W}pPr')
        if pPr is not None:
            sectPr = pPr.find(f'{W}sectPr')
            if sectPr is not None:
                # Check if the NEXT real paragraph starts with "The main intrinsic"
                for j in range(i+1, min(i+4, len(children))):
                    next_text = get_para_text(children[j])
                    if next_text.strip().startswith('The main intrinsic evaluation'):
                        figure3_break_para = child
                        figure3_text_para = children[j]
                        print(f"   Found Figure 3 transition at paragraph: {text[:60]}...")
                        break
    
    if figure3_text_para is not None and figure3_break_para is not None:
        # Move the section break from figure3_break_para to AFTER figure3_text_para
        # This keeps the text paragraph in the 2-col section
        
        pPr_src = figure3_break_para.find(f'{W}pPr')
        sectPr_src = pPr_src.find(f'{W}sectPr')
        pPr_src.remove(sectPr_src)
        
        # Add section break to figure3_text_para
        pPr_dst = figure3_text_para.find(f'{W}pPr')
        if pPr_dst is None:
            pPr_dst = ET.SubElement(figure3_text_para, f'{W}pPr')
            figure3_text_para.remove(pPr_dst)
            figure3_text_para.insert(0, pPr_dst)
        pPr_dst.append(sectPr_src)
        print("   Moved section break: body text now stays in 2-col section")
    else:
        print("   (Figure 3 text paragraph not found, skipping)")

    # ===== 5. Fix column spacing consistency =====
    print("5. Ensuring consistent column spacing...")
    for sectPr in doc_root.iter(f'{W}sectPr'):
        cols = sectPr.find(f'{W}cols')
        if cols is not None:
            num = cols.get(f'{W}num', '1')
            if num == '2':
                cols.set(f'{W}space', '480')  # Match template's 2-col spacing

    # ===== Write output =====
    print("\n6. Writing fixed document...")
    doc_bytes = ET.tostring(doc_root, xml_declaration=True, encoding='UTF-8')
    styles_bytes = ET.tostring(styles_root, xml_declaration=True, encoding='UTF-8')

    all_files['word/document.xml'] = doc_bytes
    all_files['word/styles.xml'] = styles_bytes

    # Remove old output and write new zip
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    with zipfile.ZipFile(OUTPUT_PATH, 'w', zipfile.ZIP_DEFLATED) as zout:
        for name, data in all_files.items():
            zout.writestr(name, data)

    print(f"\nDone! Saved to: {OUTPUT_PATH}")
    print(f"File size: {os.path.getsize(OUTPUT_PATH) / 1024:.0f} KB")


if __name__ == '__main__':
    process_document()
