"""Validacion post-ejecucion: comprueba que no hay celdas con error y cuenta figuras/tablas."""
import json
from pathlib import Path

NB_DIR = Path(__file__).parent

for nb_name in ["analisis_entrenamiento_hasta_e2_1.ipynb", "analisis_inferencias_progresivas.ipynb"]:
    p = NB_DIR / nb_name
    if not p.exists():
        print(f"{nb_name}: NO EXISTE"); continue
    nb = json.loads(p.read_text())
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    errors = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        for out in c.get("outputs", []):
            if out.get("output_type") == "error":
                errors.append((i, out.get("ename"), out.get("evalue")))
    executed = sum(1 for c in nb["cells"] if c["cell_type"] == "code" and c.get("execution_count") is not None)
    print(f"\n=== {nb_name} ===")
    print(f"  Celdas: {len(nb['cells'])} ({n_code} codigo, {n_md} markdown)")
    print(f"  Celdas de codigo ejecutadas: {executed}/{n_code}")
    print(f"  Errores encontrados: {len(errors)}")
    for i, ename, evalue in errors[:20]:
        print(f"    - celda {i}: {ename}: {str(evalue)[:200]}")

for figdir in ["figuras_entrenamiento_hasta_e2_1", "figuras_inferencias_progresivas"]:
    d = NB_DIR / figdir
    n = len(list(d.glob("*.png"))) if d.exists() else 0
    print(f"\n{figdir}: {n} figuras PNG")

for tabdir in ["resultados_entrenamiento_hasta_e2_1", "resultados_inferencias_progresivas"]:
    d = NB_DIR / tabdir
    n = len(list(d.glob("*.csv"))) if d.exists() else 0
    print(f"{tabdir}: {n} tablas CSV")
