from __future__ import annotations

from pathlib import Path

from scripts.export_v5_paper_tables import export_all


def test_step5_paper_tables_and_figures_export(tmp_path: Path) -> None:
    outputs = export_all(out_dir=tmp_path)
    tex_path = Path(outputs["latex_tables"])
    md_path = Path(outputs["markdown_tables"])

    assert tex_path.exists()
    assert md_path.exists()
    tex_text = tex_path.read_text(encoding="utf-8")
    assert "\\begin{table*}" in tex_text
    assert "tab:v5_r1_overall" in tex_text
    assert "tab:v5_r5_ablations" in tex_text

    assert len(outputs["figures"]) == 3
    for fig in outputs["figures"]:
        assert Path(fig).exists()
        assert Path(fig).stat().st_size > 10_000
