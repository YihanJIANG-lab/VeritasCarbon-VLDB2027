import re

infile = '_pandoc_raw_fixed.tex'
outfile = 'vldb2027.tex'

with open(infile, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find \begin{document}
doc_start = None
for i, line in enumerate(lines):
    if line.strip() == r'\begin{document}':
        doc_start = i
        break

body_lines = lines[doc_start+1:]

# === Preamble ===
preamble = r'''\PassOptionsToPackage{unicode}{hyperref}
\PassOptionsToPackage{hyphens}{url}
\documentclass[9pt,twocolumn]{extarticle}
\usepackage[letterpaper, top=1in, bottom=1in, left=0.75in, right=0.75in]{geometry}
\usepackage{times}
\usepackage{mathptmx}
\usepackage{helvet}
\renewcommand{\familydefault}{\rmdefault}
\usepackage{xcolor}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{multirow}
\usepackage{calc}
\usepackage{etoolbox}
\usepackage{url}
\usepackage{hyperref}
\usepackage{bookmark}
\usepackage{setspace}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
\usepackage{enumitem}
\setlist{nosep,leftmargin=*}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3pt plus 1pt minus 1pt}
\setlength{\emergencystretch}{3em}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\urlstyle{same}
\hypersetup{hidelinks,pdfcreator={LaTeX via pandoc}}

\title{\textbf{\sffamily\fontsize{18}{22}\selectfont VeritasCarbon: Traceable Instruction Data Generation for ESG}}
\author{%
\normalsize
\textbf{Yihan Jiang}\\ Society Hub, HKUST(GZ)\\ yjiang909@connect.hkust-gz.edu.cn
\and
\textbf{Fei Peng}\\ Nanchang University\\ pengfei24@email.ncu.edu.cn
\and
\textbf{Kok Sin Woon}\\ Society Hub, HKUST(GZ)\\ koksinwoon@hkust-gz.edu.cn
\and
\textbf{Qianping Ren}\\ Society Hub, HKUST(GZ)\\ qianpingren@hkust-gz.edu.cn
\and
\textbf{Yichang Xu}\\ Sun Yat-sen University\\ xuych73@mail2.sysu.edu.cn
\and
\textbf{Yujing Yang}\\ Sun Yat-sen University\\ Yangyj256@mail2.sysu.edu.cn
}
\date{\vspace{-2ex}}

\begin{document}
\maketitle
\begin{abstract}
'''

# === Process body line by line ===
output = []
in_author_table = False

for i, line in enumerate(body_lines):
    s = line.strip()
    
    # Skip author table (from \begin{longtable} to \end{longtable} and the closing })
    if s.startswith(r'\begin{longtable}'):
        in_author_table = True
        continue
    if in_author_table and s == r'\end{longtable}':
        in_author_table = False
        # Also skip the next line if it's just "}"
        if i + 1 < len(body_lines) and body_lines[i+1].strip() == '}':
            # Mark next line to be skipped by setting a flag or using index
            # But we can't easily skip the next line here. Instead, we'll handle it below.
            pass
        continue
    if in_author_table:
        continue
    
    # Skip ABSTRACT bold line
    if s == r'\textbf{ABSTRACT}':
        continue
    
    # Skip orphan table lines and standalone braces from author table
    if s in ['', '&', r'\\', r'\toprule\noalign{}', r'\bottomrule\noalign{}', 
             r'\endhead', r'\endlastfoot', r'\midrule\noalign{}', '}']:
        continue
    
    # Section headers
    m = re.match(r'^\\textbf\{(\d)\}\s*\u2003\s*([A-Z][A-Z\s\-,/]+)$', s)
    if m:
        output.append(f'\\section{{{m.group(2)}}}\n')
        continue
    
    m = re.match(r'^(\d)\s*\u2003\s*([A-Z][A-Z\s\-,/]+)$', s)
    if m:
        output.append(f'\\section{{{m.group(2)}}}\n')
        continue
    
    if s == r'\textbf{APPENDIX}':
        output.append('\\section{APPENDIX}\n')
        continue
    
    if s == 'REFERENCES`':
        output.append('\\section{References}\n')
        continue
    
    # Subsection headers
    m = re.match(r'^\\textbf\{(\d\.\d)\}\s*\u2003\s*\\textbf\{(.+)\}$', s)
    if m:
        output.append(f'\\subsection{{{m.group(2)}}}\n')
        continue
    
    m = re.match(r'^(\d\.\d)\s*\u2003\s*(.+)$', s)
    if m:
        output.append(f'\\subsection{{{m.group(2)}}}\n')
        continue
    
    m = re.match(r'^\\textbf\{(\d\.\d)\}\s+(.+)$', s)
    if m:
        output.append(f'\\subsection{{{m.group(2)}}}\n')
        continue
    
    # Subsubsection headers
    m = re.match(r'^\\textbf\{(A\.\d\.\d)\}\s+(.+)$', s)
    if m:
        output.append(f'\\subsubsection{{{m.group(2)}}}\n')
        continue
    
    m = re.match(r'^\\textbf\{(\d\.\d\.\d)\}\s+(.+)$', s)
    if m:
        output.append(f'\\subsubsection{{{m.group(2)}}}\n')
        continue
    
    # Appendix subsections
    m = re.match(r'^\\textbf\{(A\.\d)\}\s+(.+)$', s)
    if m:
        output.append(f'\\subsection{{{m.group(2)}}}\n')
        continue
    
    output.append(line)

body = ''.join(output)

# Close abstract
body = body.replace(
    'Copyright 2027 VLDB Endowment 2150-8097/27/XX \\ldots{} \\$10.00.\n',
    'Copyright 2027 VLDB Endowment 2150-8097/27/XX \\ldots{} \\$10.00.\n\\end{abstract}\n'
)

# Fix image paths
body = body.replace('./media/media/image1.emf', '../results/figures_and_tables/figure1.png')
body = body.replace('./media/media/image2.emf', '../results/figures_and_tables/figure2.png')
body = body.replace('./media/media/image3.png', '../results/figures_and_tables/fig5_coe_vs_baselines.png')
body = body.replace('./media/media/image4.png', '../results/scalability/fig_scalability_corpus_size.png')
body = body.replace('./media/media/image5.png', '../results/figures_and_tables/fig2_quality_distribution.png')
body = body.replace('./media/media/image6.emf', '../results/figures_and_tables/fig1_expert_usage.png')

# Fix figures using regex for more robust matching
import re as re_mod

# Figure 1
body = re_mod.sub(
    r'\\includegraphics\[width=7in,height=4\.38472in\]\{\.\./results/figures_and_tables/figure1\.png\}\s*'
    r'\\protect\\phantomsection\\label\{_Ref226683296\}\{\}Figure\s+1\s+The\s+VeritasCarbon\s*\n'
    r'pipeline\.\s+Raw\s+ESG\s+documents\s+are\s+processed\s+through\s+four\s+stages\s+to\s+produce\s*\n'
    r'traceable\s+instruction-response\s+pairs',
    r'\\begin{figure*}[htbp]\n\\centering\n\\includegraphics[width=\\textwidth]{../results/figures_and_tables/figure1.png}\n\\caption{The VeritasCarbon pipeline. Raw ESG documents are processed through four stages to produce traceable instruction-response pairs.}\n\\label{fig:1}\n\\end{figure*}',
    body
)

# Figure 2
body = re_mod.sub(
    r'\\includegraphics\[width=7in,height=1\.76111in\]\{\.\./results/figures_and_tables/figure2\.png\}\s*'
    r'\\protect\\phantomsection\\label\{_Ref227987336\}\{\}Figure\s+2\s+CoDE\s+internal\s*\n'
    r'architecture\.\s+Layered\s+expert\s+selection,\s+MetaExpert\s+orchestration,\s*\n'
    r'collaboration\s+modes,\s+and\s+feedback\s+refinement',
    r'\\begin{figure*}[htbp]\n\\centering\n\\includegraphics[width=\\textwidth]{../results/figures_and_tables/figure2.png}\n\\caption{CoDE internal architecture. Layered expert selection, MetaExpert orchestration, collaboration modes, and feedback refinement.}\n\\label{fig:2}\n\\end{figure*}',
    body
)

# Figure 4 (baselines) - in original doc it's "Figure 4"
body = re_mod.sub(
    r'\\includegraphics\[width=3\.24931in,height=2\.43403in\]\{\.\./results/figures_and_tables/fig5_coe_vs_baselines\.png\}\s*'
    r'Figure\s+4\s+Comparison\s+of\s+CoDE\s+against\s+three\s+baselines',
    r'\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=\\columnwidth]{../results/figures_and_tables/fig5_coe_vs_baselines.png}\n\\caption{Comparison of CoDE against three baselines.}\n\\label{fig:4}\n\\end{figure}',
    body
)

# Figure 5 (scalability)
body = re_mod.sub(
    r'\\includegraphics\[width=3\.25in,height=1\.3625in\]\{\.\./results/scalability/fig_scalability_corpus_size\.png\}\s*'
    r'\\protect\\phantomsection\\label\{_Ref226685825\}\{\}Figure\s+5\s+Generation\s+time\s*\n'
    r'scales\s+linearly\s+with\s+corpus\s+size\.\s+\(a\)\s+Scalability\s+of\s+CoDE;\s+\(b\)\s*\n'
    r'throughput\s+stability',
    r'\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=\\columnwidth]{../results/scalability/fig_scalability_corpus_size.png}\n\\caption{Generation time scales linearly with corpus size. (a) Scalability of CoDE; (b) throughput stability.}\n\\label{fig:5}\n\\end{figure}',
    body
)

# Figure 6 (quality distribution)
body = re_mod.sub(
    r'\\includegraphics\[width=3\.25in,height=1\.26875in\]\{\.\./results/figures_and_tables/fig2_quality_distribution\.png\}\s*'
    r'\\protect\\phantomsection\\label\{_Ref227974620\}\{\}Figure\s+6\s+Distribution\s+of\s*\n'
    r'expert\s+agent\s+invocations\s+across\s+35,009\s+QA\s+pairs\.\s+Summary\s+Expert\s*\n'
    r'dominates\s+base-layer\s+usage,\s+while\s+Consistency\s+Verification\s+Expert\s+is\s+the\s*\n'
    r'most\s+frequently\s+activated\s+upper-layer\s+agent',
    r'\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=\\columnwidth]{../results/figures_and_tables/fig2_quality_distribution.png}\n\\caption{Distribution of quality scores across 35,009 generated QA pairs.}\n\\label{fig:6}\n\\end{figure}',
    body
)

# Figure 7 (expert usage)
body = re_mod.sub(
    r'\\includegraphics\[width=3\.25in,height=2\.03125in\]\{\.\./results/figures_and_tables/fig1_expert_usage\.png\}\s*'
    r'\\protect\\phantomsection\\label\{_Ref226558583\}\{\}\\textbf\{Figure\s+7:\}\s*\n'
    r'Distribution\s+of\s+quality\s+scores\s+across\s+35,009\s+generated\s+QA\s+pairs\.\s+The\s*\n'
    r'distribution\s+centers\s+around\s+0\.667\s+with\s+\\\(\\mathbf\{\\sigma\}\\\)\s+=\s+0\.103,\s+and\s*\n'
    r'89\.7\\%\s+of\s+pairs\s+exceed\s+the\s+minimum\s+threshold\s+\\\(\\mathbf\{\\tau\}\\\)\s+=\s*\n'
    r'0\\textbf\{\.5\.\}',
    r'\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=\\columnwidth]{../results/figures_and_tables/fig1_expert_usage.png}\n\\caption{Distribution of expert agent invocations across 35,009 QA pairs. Summary Expert dominates base-layer usage, while Consistency Verification Expert is the most frequently activated upper-layer agent.}\n\\label{fig:7}\n\\end{figure}',
    body
)

# Fix cross-references
body = body.replace('\\hyperref[_Ref226683296]{Figure 1}', 'Figure~\\ref{fig:1}')
body = body.replace('\\hyperref[_Ref227987336]{Figure 2}', 'Figure~\\ref{fig:2}')
body = body.replace('\\hyperref[_Ref226685825]{Figure 4}', 'Figure~\\ref{fig:5}')
body = body.replace('\\hyperref[_Ref227974620]{Figure 5}', 'Figure~\\ref{fig:6}')
body = body.replace('\\hyperref[_Ref226558583]{Figure 6}', 'Figure~\\ref{fig:7}')
body = body.replace('\\hyperref[_Ref227939794]{Table 2}', 'Table~\\ref{tab:2}')

# Fix tables: remove LTcaptype wrappers
body = re_mod.sub(r'\{\\def\\LTcaptype\{none\}\s*% do not increment counter\s*\n', '', body)

# Fix tables: convert longtable to table+tabular
def replace_longtable(match):
    full = match.group(0)
    m = re_mod.search(r'\\begin\{longtable\}\[\]\{(.*?)\}', full)
    if not m:
        return full
    colspec = m.group(1)
    content = full
    content = re_mod.sub(r'\\begin\{longtable\}\[\]\{.*?\}\s*', '', content)
    content = content.replace('\\end{longtable}', '')
    content = content.replace('\\endhead', '')
    content = content.replace('\\endlastfoot', '')
    content = re_mod.sub(r'\\(top|bottom|mid)rule\\noalign\{\}', '', content)
    content = content.strip()
    return f"""\\begin{{table}}[htbp]
\\centering
\\small
\\begin{{tabular}}{{{colspec}}}
{content}
\\end{{tabular}}
\\end{{table}}"""

body = re_mod.sub(r'\\begin\{longtable\}\[\]\{.*?\}.*?\\end\{longtable\}', replace_longtable, body, flags=re_mod.DOTALL)

# Clean orphan braces after tables
body = body.replace('\\end{table}\n\n}\n\nNotes.', '\\end{table}\n\nNotes.')
body = re_mod.sub(r'\\end\{table\}\n\n\}\n(?=\\w|\\section|\\subsection|\\begin|\\textbf)', r'\\end{table}\n\n', body)

# Add table label
if 'Table~\\ref{tab:2} Corpus scope' in body:
    body = body.replace('Table~\\ref{tab:2} Corpus scope', '\\label{tab:2}\nTable~\\ref{tab:2} Corpus scope')

# Fix formula
body = body.replace(
    'If \\textbar{\\(s_{i} \\geq 0.5\\)}\\textbar{} ≥ 2,',
    'If $\\{s_i \\geq 0.5\\} \\geq 2$,'
)

# Fix Unicode characters
body = body.replace('τ', r'$\tau$')
body = body.replace('𝑛', '$n$')

# Remove extra \end{document}
body = body.replace('\\end{document}\n\\end{document}', '\\end{document}')

with open(outfile, 'w', encoding='utf-8') as f:
    f.write(preamble)
    f.write(body)

print(f"Wrote {outfile}")
