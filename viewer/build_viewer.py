"""
Build the self-contained pose viewer page.

Usage:
    python viewer/build_viewer.py [output.html]

Default output: viewer/pose_viewer.html. The page inlines the etho-pose importer
(viewer/pose_format.js), the PDF report builder (viewer/report_pdf.js), the Python API shipped by "Export script"
(api/etho.py) and the analysis-notebook template (viewer/analysis_notebook.py, split into
cells), and loads three.js from a CDN (jsPDF too, on the first PDF export). No data is bundled: use the page's "Load data" button (or drop
a file on it) to open an etho-pose file or a legacy pose-testset JSON such as examples/mouse_3d.json. The
formats are described in viewer/POSE_FORMAT.md and viewer/CLUSTER_FORMAT.md.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def notebook_cells(path):
    """Split a Jupytext percent-format file into [{type, source}] cells (test-only cells skipped)."""
    cells, cur = [], None
    for line in path.read_text(encoding="utf-8").split('\n'):
        if line.startswith('# %%'):
            if cur:
                cells.append(cur)
            cur = {'type': 'markdown' if '[markdown]' in line else 'code', 'lines': []}
            continue
        if cur is None:
            continue                       # header comment before the first cell
        if cur['type'] == 'markdown':
            line = line[2:] if line.startswith('# ') else line.lstrip('#')
        cur['lines'].append(line)
    if cur:
        cells.append(cur)
    out = []
    for c in cells:
        src = '\n'.join(c['lines']).strip('\n')
        if not src or src.startswith('# (test shim: not exported)'):
            continue
        out.append({'type': c['type'], 'source': src})
    return out


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "pose_viewer.html"
    importer = (HERE / "pose_format.js").read_text(encoding="utf-8")
    assert "</" not in importer  # safe to inline inside <script>
    report = (HERE / "report_pdf.js").read_text(encoding="utf-8")
    assert "</" not in report
    cleaner = (HERE / "pose_clean.js").read_text(encoding="utf-8")
    assert "</" not in cleaner
    cells = json.dumps(notebook_cells(HERE / "analysis_notebook.py"))
    assert "</" not in cells     # safe to inline inside <script type="application/json">
    etho_py = json.dumps((HERE.parent / "api" / "etho.py").read_text(encoding="utf-8"))
    assert "</" not in etho_py   # safe to inline inside <script type="application/json">
    template = (HERE / "viewer_template.html").read_text(encoding="utf-8")
    assert template.count("/*__POSE_FORMAT__*/") == 1
    assert template.count("/*__NOTEBOOK__*/") == 1
    assert template.count("/*__REPORT__*/") == 1
    assert template.count("/*__POSE_CLEAN__*/") == 1
    assert template.count("/*__ETHO_PY__*/") == 1
    out.write_text(template.replace("/*__POSE_FORMAT__*/", importer).replace("/*__REPORT__*/", report).replace("/*__POSE_CLEAN__*/", cleaner)
                   .replace("/*__NOTEBOOK__*/", cells).replace("/*__ETHO_PY__*/", etho_py), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
