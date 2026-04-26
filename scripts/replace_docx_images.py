"""Replace screenshot images in docx with high-quality originals."""
import zipfile
import shutil
import os

docx_path = 'docs/2026-04-10-VeritasCarbon Traceable Instruction D.docx'
output_path = 'docs/2026-04-10-VeritasCarbon Traceable Instruction D_replaced.docx'

# Mapping: docx image name -> replacement file path
replacements = {
    'word/media/image3.png': 'results/figures_and_tables/fig_baseline_comparison_6panel.png',  # Fig 3 baseline comparison
    'word/media/image4.png': 'results/scalability/fig_scalability_corpus_size.png',            # Fig 4 scalability
    'word/media/image5.png': 'paper/figures/fig1_expert_usage.png',                            # Fig 6 expert usage
}

# Create output docx by copying and replacing images
with zipfile.ZipFile(docx_path, 'r') as zin:
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename in replacements:
                # Replace with high-quality original
                replacement_path = replacements[item.filename]
                with open(replacement_path, 'rb') as f:
                    data = f.read()
                zout.writestr(item, data)
                orig_size = os.path.getsize(replacement_path)
                print(f'REPLACED: {item.filename} <- {replacement_path} ({orig_size} bytes)')
            else:
                # Copy as-is
                data = zin.read(item.filename)
                zout.writestr(item, data)

print(f'\nOutput: {output_path}')
print(f'Size: {os.path.getsize(output_path)} bytes')
