"""
Remove section breaks #4 and #5 around Figure 3 so that:
- Text flows naturally in 2-column layout (fill left first, then right)
- Figure 3 displays inline within the 2-column flow
- No more column balancing at the section break point
"""
import zipfile
import os
from xml.etree import ElementTree as ET

# Register namespaces
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

INPUT_PATH = 'docs/2026-04-10-VeritasCarbon Traceable Instruction D_formatted.docx'
OUTPUT_PATH = 'docs/2026-04-10-VeritasCarbon Traceable Instruction D_formatted.docx'

with zipfile.ZipFile(INPUT_PATH, 'r') as zin:
    all_files = {name: zin.read(name) for name in zin.namelist()}

doc_root = ET.fromstring(all_files['word/document.xml'])
body = doc_root.find(f'.//w:body', ns)

children = list(body)
removed = 0
break_count = 0

for i, child in enumerate(children):
    if child.tag != f'{W}p':
        continue
    
    pPr = child.find(f'{W}pPr')
    if pPr is None:
        continue
    
    sectPr = pPr.find(f'{W}sectPr')
    if sectPr is None:
        continue
    
    break_count += 1
    text = ''.join(t.text or '' for t in child.iter(f'{W}t'))
    cols = sectPr.find(f'{W}cols')
    col_num = cols.get(f'{W}num', '1') if cols is not None else '1'
    
    # Break #4: cols=2, text starts with "The main intrinsic evaluation"
    # Break #5: cols=1, empty paragraph after Figure 3 caption
    if break_count == 4:
        pPr.remove(sectPr)
        removed += 1
        print(f"Removed break #{break_count} (cols={col_num}) from para [{i}]: {text[:60]}...")
    elif break_count == 5:
        pPr.remove(sectPr)
        removed += 1
        print(f"Removed break #{break_count} (cols={col_num}) from para [{i}]: {text[:60] if text.strip() else '[empty]'}...")

print(f"\nRemoved {removed} section breaks total")

# Write output
doc_bytes = ET.tostring(doc_root, xml_declaration=True, encoding='UTF-8')
all_files['word/document.xml'] = doc_bytes

if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)

with zipfile.ZipFile(OUTPUT_PATH, 'w', zipfile.ZIP_DEFLATED) as zout:
    for name, data in all_files.items():
        zout.writestr(name, data)

print(f"Saved: {OUTPUT_PATH}")

# Verify remaining section structure
print("\n--- Remaining sections ---")
doc_root2 = ET.fromstring(doc_bytes)
for i, sectPr in enumerate(doc_root2.iter(f'{W}sectPr')):
    cols = sectPr.find(f'{W}cols')
    col_num = cols.get(f'{W}num', '1') if cols is not None else '1'
    col_space = cols.get(f'{W}space', 'N/A') if cols is not None else 'N/A'
    print(f"  Section {i+1}: cols={col_num} space={col_space}")
