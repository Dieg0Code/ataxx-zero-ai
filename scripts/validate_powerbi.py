"""Validar el reporte PBIR: confirma que cada referencia de campo en los visuales
existe como columna o medida en la tabla del modelo. También revisa visualTypes
contra una lista conocida.

Uso:
    uv run python scripts/validate_powerbi.py

Sale con código 1 si encuentra problemas.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO / "runs_history" / "ataxx_zero.pbip"
REPORT_DIR = REPO / "runs_history" / "ataxx_zero.Report"
MODEL_DIR = REPO / "runs_history" / "ataxx_zero.SemanticModel"
PAGES_DIR = REPORT_DIR / "definition" / "pages"
TABLES_DIR = MODEL_DIR / "definition" / "tables"

# Whitelist estricta: tipos que sabemos que renderizan bien en este proyecto
# (los mismos que usa la referencia ink_pulse_studio).
VALID_VISUAL_TYPES = {
    "card",
    "tableEx",
    "barChart",
    "lineChart",
    "donutChart",
    "pieChart",
    "slicer",
    "textbox",
}

# Campos en objects.* que la referencia evita en visuales individuales y que
# pueden romper el renderizado si se aplican a cards/lineCharts/etc.
PROHIBITED_OBJECT_KEYS = {
    "background",
    "border",
}

# Sub-propiedades sospechosas dentro de objects.*.properties (a cualquier nivel)
SUSPICIOUS_SUBPROPS = {
    "transparency",
    "radius",
    "gridlineColor",
}

# Keys válidos en objects.* (los que la referencia testea y que sabemos que
# renderizan). Cualquier otro campo va a hacer fallar la visual entera con
# "Hubo un problema con uno o más campos".
ALLOWED_OBJECT_KEYS = {
    "general", "title", "labels", "categoryLabels", "categoryAxis", "valueAxis",
    "legend", "dataPoint", "data", "header", "items", "columnHeaders", "values",
    "grid",
}


def parse_table_tmdl(path: Path) -> dict:
    """Devuelve dict con name, columns (set) y measures (set) de un .tmdl table."""
    text = path.read_text(encoding="utf-8")
    name_match = re.match(r"\s*table\s+(\S+)", text)
    name = name_match.group(1) if name_match else path.stem
    columns: set[str] = set()
    measures: set[str] = set()
    # column <name>
    for match in re.finditer(r"^\s*column\s+'?([^'\n]+?)'?\s*$", text, re.MULTILINE):
        columns.add(match.group(1).strip())
    # measure '<name>' = ...
    for match in re.finditer(r"^\s*measure\s+'?([^'=]+?)'?\s*=", text, re.MULTILINE):
        measures.add(match.group(1).strip())
    return {"name": name, "columns": columns, "measures": measures}


def load_model_tables() -> dict[str, dict]:
    tables: dict[str, dict] = {}
    for tmdl in TABLES_DIR.glob("*.tmdl"):
        info = parse_table_tmdl(tmdl)
        tables[info["name"]] = info
    return tables


def walk_visuals():
    for page_dir in sorted(PAGES_DIR.iterdir()):
        if not page_dir.is_dir():
            continue
        visuals_dir = page_dir / "visuals"
        if not visuals_dir.is_dir():
            continue
        page_name = page_dir.name
        for visual_dir in sorted(visuals_dir.iterdir()):
            visual_path = visual_dir / "visual.json"
            if visual_path.is_file():
                yield page_name, visual_path


def _find_suspicious_subprops(node, found: set[str] | None = None) -> set[str]:
    """Recursively scan an objects dict for keys in SUSPICIOUS_SUBPROPS."""
    if found is None:
        found = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key in SUSPICIOUS_SUBPROPS:
                found.add(key)
            _find_suspicious_subprops(value, found)
    elif isinstance(node, list):
        for item in node:
            _find_suspicious_subprops(item, found)
    return found


def collect_field_references(node, refs: list[tuple[str, str, str]]) -> None:
    """Walks a query state JSON tree and collects (kind, entity, property)."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "field" and isinstance(value, dict):
                if "Column" in value:
                    col = value["Column"]
                    entity = col.get("Expression", {}).get("SourceRef", {}).get("Entity", "?")
                    prop = col.get("Property", "?")
                    refs.append(("column", entity, prop))
                elif "Measure" in value:
                    meas = value["Measure"]
                    entity = meas.get("Expression", {}).get("SourceRef", {}).get("Entity", "?")
                    prop = meas.get("Property", "?")
                    refs.append(("measure", entity, prop))
            else:
                collect_field_references(value, refs)
    elif isinstance(node, list):
        for item in node:
            collect_field_references(item, refs)


def main() -> int:
    if not PROJECT.is_file():
        print(f"[error] no existe el proyecto PBIP en {PROJECT}", file=sys.stderr)
        return 1

    tables = load_model_tables()
    if not tables:
        print(f"[error] no encontré tablas en {TABLES_DIR}", file=sys.stderr)
        return 1

    problems: list[str] = []

    # Check page.json files have no objects (matches reference convention).
    for page_dir in sorted(PAGES_DIR.iterdir()):
        if not page_dir.is_dir():
            continue
        page_json_path = page_dir / "page.json"
        if not page_json_path.is_file():
            continue
        try:
            page_data = json.loads(page_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"  [JSON inválido en página] {page_dir.name}/page.json: {exc}")
            continue
        if "objects" in page_data:
            problems.append(f"  [campo prohibido en página] {page_dir.name}/page.json tiene 'objects' (la referencia no lo usa)")

    visuals_seen = 0
    for page_name, visual_path in walk_visuals():
        visuals_seen += 1
        try:
            data = json.loads(visual_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"  [JSON inválido]   {page_name}/{visual_path.parent.name}: {exc}")
            continue

        visual = data.get("visual", {})
        vtype = visual.get("visualType")
        if vtype and vtype not in VALID_VISUAL_TYPES:
            problems.append(f"  [visualType inválido]   {page_name}/{visual_path.parent.name}: '{vtype}' (válidos: {sorted(VALID_VISUAL_TYPES)})")

        # Detectar objects.background/border (la referencia nunca los usa en visuals)
        objs = visual.get("objects", {})
        for forbidden in PROHIBITED_OBJECT_KEYS:
            if forbidden in objs:
                problems.append(f"  [campo prohibido]      {page_name}/{visual_path.parent.name}: objects.{forbidden} no debe estar en visuals individuales")

        # Detectar objects.* fuera de la whitelist probada
        for key in objs:
            if key not in ALLOWED_OBJECT_KEYS and key not in PROHIBITED_OBJECT_KEYS:
                problems.append(f"  [objects.{key} desconocido] {page_name}/{visual_path.parent.name}: no aparece en la referencia, puede romper la visual")

        # Detectar sub-props sospechosas en cualquier rama de objects
        susp_found = _find_suspicious_subprops(objs)
        for sp in susp_found:
            problems.append(f"  [campo sospechoso]     {page_name}/{visual_path.parent.name}: objects.*.{sp} no aparece en la referencia, puede romper el render")

        refs: list[tuple[str, str, str]] = []
        collect_field_references(visual, refs)
        for kind, entity, prop in refs:
            table = tables.get(entity)
            if table is None:
                problems.append(f"  [tabla inexistente]   {page_name}/{visual_path.parent.name}: refiere tabla '{entity}'")
                continue
            if kind == "column" and prop not in table["columns"]:
                problems.append(f"  [columna inexistente] {page_name}/{visual_path.parent.name}: '{entity}.{prop}' no existe (tiene: {sorted(table['columns'])[:5]}…)")
            elif kind == "measure" and prop not in table["measures"]:
                problems.append(f"  [medida inexistente]  {page_name}/{visual_path.parent.name}: '{entity}.{prop}' no existe (tiene: {sorted(table['measures'])[:5]}…)")

    print(f"Validados {visuals_seen} visuales en {len(tables)} tablas.")
    if problems:
        print(f"\n{len(problems)} problemas:\n")
        for p in problems:
            print(p)
        return 1
    print("OK — todos los visuales referencian campos válidos y tipos válidos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
