"""
Build the self-contained pose viewer page.

Usage:
    python viewer/build_viewer.py [output.html]

Default output: viewer/pose_viewer.html. The page inlines the etho-pose importer
(viewer/pose_format.js) and loads three.js from a CDN. No data is bundled: use the page's
"Load data" button (or drop a file on it) to open an etho-pose file or a legacy
pose-testset JSON such as mouse_3d.json. The format is described in viewer/POSE_FORMAT.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "pose_viewer_v0.html"
    importer = (HERE / "pose_format.js").read_text()
    assert "</" not in importer  # safe to inline inside <script>
    template = (HERE / "viewer_template.html").read_text()
    assert template.count("/*__POSE_FORMAT__*/") == 1
    out.write_text(template.replace("/*__POSE_FORMAT__*/", importer))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
