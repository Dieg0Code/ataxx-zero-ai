"""Genera las páginas del reporte PBIR de ataxx_zero, clonando 1:1 los patrones
del proyecto `ink_pulse_studio` (referencia que sabemos que funciona).

Reglas estrictas:
- Sin `objects.background`, `objects.border`, `objects.transparency`, `objects.radius`
  en visuales individuales.
- Cada chart con `objects.dataPoint.fill` para que la serie tenga color.
- `page.json` mínimo (sin `objects`).
- VisualTypes restringidos a los que la referencia testea: card, barChart,
  lineChart, tableEx, slicer, textbox.

Uso:
    uv run python scripts/build_powerbi_pages.py
    uv run python scripts/validate_powerbi.py   # antes de abrir Power BI
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO / "runs_history" / "ataxx_zero.Report"
PAGES_DIR = REPORT_DIR / "definition" / "pages"
TABLE = "all_runs"

# Paleta sobria (matched al proyecto de referencia)
TITLE = "#0E2A43"
TEXT = "#1F2937"
TEXT_DIM = "#475569"
SUBTITLE = "#334155"
ACCENT_RED = "#E31B2F"
ACCENT_BLUE = "#1F4E79"
ACCENT_DARK = "#0E2A43"
GRID_OUTLINE = "#D7E4EE"
HEADER_BACK = "#EAF2F8"

VISUAL_SCHEMA = "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/visualContainer/2.8.0/schema.json"
PAGE_SCHEMA = "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/page/2.1.0/schema.json"
PAGES_META_SCHEMA = "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/pagesMetadata/1.0.0/schema.json"

PAGE_W = 1280
PAGE_H = 720


# ---------- low-level helpers (mirror reference exactly) ----------

def lit(value):
    if isinstance(value, bool):
        return {"expr": {"Literal": {"Value": "true" if value else "false"}}}
    if isinstance(value, (int, float)):
        return {"expr": {"Literal": {"Value": f"{value}D"}}}
    return {"expr": {"Literal": {"Value": f"'{value}'"}}}


def solid(hex_color):
    return {"solid": {"color": lit(hex_color)}}


def col_field(prop, table=TABLE):
    return {
        "field": {"Column": {"Expression": {"SourceRef": {"Entity": table}}, "Property": prop}},
        "queryRef": f"{table}.{prop}",
        "nativeQueryRef": prop,
        "active": True,
    }


def meas_field(prop, table=TABLE, active=False):
    field = {
        "field": {"Measure": {"Expression": {"SourceRef": {"Entity": table}}, "Property": prop}},
        "queryRef": f"{table}.{prop}",
        "nativeQueryRef": prop,
    }
    if active:
        field["active"] = True
    return field


# ---------- visual blueprints (clones of reference) ----------

def textbox(name, x, y, w, h, z, paragraphs):
    """paragraphs: list of dicts with keys text, size (e.g. '11pt'), bold, color, align."""
    return {
        "$schema": VISUAL_SCHEMA,
        "name": name,
        "position": {"x": x, "y": y, "z": z, "height": h, "width": w, "tabOrder": z},
        "visual": {
            "visualType": "textbox",
            "objects": {
                "general": [
                    {
                        "properties": {
                            "paragraphs": [
                                {
                                    "textRuns": [
                                        {
                                            "value": p["text"],
                                            "textStyle": {
                                                "fontSize": p.get("size", "11pt"),
                                                "fontWeight": "bold" if p.get("bold") else "normal",
                                                "color": p.get("color", TEXT),
                                            },
                                        }
                                    ],
                                    "horizontalTextAlignment": p.get("align", "left"),
                                }
                                for p in paragraphs
                            ]
                        }
                    }
                ]
            },
            "drillFilterOtherVisuals": True,
        },
    }


def card(name, x, y, w, h, z, *, measure, value_color=ACCENT_DARK):
    return {
        "$schema": VISUAL_SCHEMA,
        "name": name,
        "position": {"x": x, "y": y, "z": z, "height": h, "width": w, "tabOrder": z},
        "visual": {
            "visualType": "card",
            "query": {"queryState": {"Values": {"projections": [meas_field(measure)]}}},
            "objects": {
                "labels": [
                    {
                        "properties": {
                            "color": solid(value_color),
                            "fontSize": lit(30),
                            "fontFamily": lit("Segoe UI Semibold"),
                        }
                    }
                ],
                "categoryLabels": [{"properties": {"show": lit(False)}}],
            },
            "drillFilterOtherVisuals": True,
        },
    }


def line_chart(name, x, y, w, h, z, *, category, y_field, legend=None, color=ACCENT_DARK,
               sort_by=None):
    """Mirror of reference line_demanda.

    category, y_field, legend: (kind, prop) tuples where kind in {"column","measure"}.
    """
    def to_field(spec):
        kind, prop = spec
        return col_field(prop) if kind == "column" else meas_field(prop)

    query_state = {
        "Category": {"projections": [to_field(category)]},
        "Y": {"projections": [to_field(y_field)]},
    }
    if legend is not None:
        query_state["Series"] = {"projections": [to_field(legend)]}

    visual = {
        "$schema": VISUAL_SCHEMA,
        "name": name,
        "position": {"x": x, "y": y, "z": z, "height": h, "width": w, "tabOrder": z},
        "visual": {
            "visualType": "lineChart",
            "query": {"queryState": query_state},
            "objects": {
                "title": [{"properties": {"show": lit(False)}}],
                "dataPoint": [{"properties": {"fill": solid(color)}}],
                "labels": [
                    {
                        "properties": {
                            "show": lit(False),
                            "fontSize": lit(9),
                            "color": solid(TEXT),
                        }
                    }
                ],
                "categoryAxis": [
                    {
                        "properties": {
                            "labelColor": solid(TEXT_DIM),
                            "fontSize": lit(10),
                        }
                    }
                ],
                "valueAxis": [
                    {
                        "properties": {
                            "labelColor": solid(TEXT_DIM),
                            "fontSize": lit(10),
                        }
                    }
                ],
            },
            "drillFilterOtherVisuals": True,
        },
    }
    if legend is not None:
        visual["visual"]["objects"]["legend"] = [
            {
                "properties": {
                    "show": lit(True),
                    "position": lit("Top"),
                    "labelColor": solid(TEXT),
                    "fontSize": lit(9),
                }
            }
        ]
    if sort_by is not None:
        skind, sprop, sdir = sort_by
        sf = col_field(sprop) if skind == "column" else meas_field(sprop)
        visual["visual"]["query"]["sortDefinition"] = {
            "sort": [{"field": sf["field"], "direction": sdir}]
        }
    return visual


def bar_chart(name, x, y, w, h, z, *, category, y_field, color=ACCENT_RED, sort_by=None,
              show_labels=True):
    """Mirror of reference bar_campanas (horizontal bars)."""
    def to_field(spec):
        kind, prop = spec
        return col_field(prop) if kind == "column" else meas_field(prop)

    visual = {
        "$schema": VISUAL_SCHEMA,
        "name": name,
        "position": {"x": x, "y": y, "z": z, "height": h, "width": w, "tabOrder": z},
        "visual": {
            "visualType": "barChart",
            "query": {
                "queryState": {
                    "Category": {"projections": [to_field(category)]},
                    "Y": {"projections": [to_field(y_field)]},
                }
            },
            "objects": {
                "title": [{"properties": {"show": lit(False)}}],
                "dataPoint": [{"properties": {"fill": solid(color)}}],
                "labels": [
                    {
                        "properties": {
                            "show": lit(show_labels),
                            "fontSize": lit(10),
                            "color": solid(TITLE),
                        }
                    }
                ],
                "categoryAxis": [
                    {
                        "properties": {
                            "labelColor": solid(SUBTITLE),
                            "fontSize": lit(10),
                        }
                    }
                ],
                "valueAxis": [
                    {
                        "properties": {
                            "labelColor": solid(SUBTITLE),
                            "fontSize": lit(10),
                        }
                    }
                ],
            },
            "drillFilterOtherVisuals": True,
        },
    }
    if sort_by is not None:
        skind, sprop, sdir = sort_by
        sf = col_field(sprop) if skind == "column" else meas_field(sprop)
        visual["visual"]["query"]["sortDefinition"] = {
            "sort": [{"field": sf["field"], "direction": sdir}]
        }
    return visual


def table_ex(name, x, y, w, h, z, *, value_specs, sort_by=None, show_total=False):
    """Mirror of reference tabla_oportunidades. tableEx wants `active: true` on
    every projection — both columns AND measures — otherwise grouping breaks."""
    def to_field(spec):
        kind, prop = spec
        return col_field(prop) if kind == "column" else meas_field(prop, active=True)

    visual = {
        "$schema": VISUAL_SCHEMA,
        "name": name,
        "position": {"x": x, "y": y, "z": z, "height": h, "width": w, "tabOrder": z},
        "visual": {
            "visualType": "tableEx",
            "query": {
                "queryState": {"Values": {"projections": [to_field(s) for s in value_specs]}}
            },
            "objects": {
                "columnHeaders": [
                    {
                        "properties": {
                            "fontSize": lit(10),
                            "fontColor": solid(TITLE),
                            "backColor": solid(HEADER_BACK),
                        }
                    }
                ],
                "values": [
                    {
                        "properties": {
                            "fontSize": lit(10),
                            "fontColorPrimary": solid(TEXT),
                        }
                    }
                ],
                "grid": [
                    {
                        "properties": {
                            "rowPadding": lit(5),
                            "outlineColor": solid(GRID_OUTLINE),
                        }
                    }
                ],
            },
            "drillFilterOtherVisuals": True,
        },
    }
    if sort_by is not None:
        skind, sprop, sdir = sort_by
        sf = col_field(sprop) if skind == "column" else meas_field(sprop)
        visual["visual"]["query"]["sortDefinition"] = {
            "sort": [{"field": sf["field"], "direction": sdir}]
        }
    return visual


def slicer(name, x, y, w, h, z, *, prop, header_text):
    """Mirror of reference slicer_artista (Dropdown mode)."""
    return {
        "$schema": VISUAL_SCHEMA,
        "name": name,
        "position": {"x": x, "y": y, "z": z, "height": h, "width": w, "tabOrder": z},
        "visual": {
            "visualType": "slicer",
            "query": {"queryState": {"Values": {"projections": [col_field(prop)]}}},
            "objects": {
                "data": [{"properties": {"mode": lit("Dropdown")}}],
                "general": [{"properties": {}}],
                "header": [
                    {
                        "properties": {
                            "show": lit(True),
                            "text": lit(header_text),
                            "fontColor": solid(TITLE),
                            "textSize": lit(11),
                        }
                    }
                ],
                "items": [
                    {
                        "properties": {
                            "textSize": lit(11),
                            "fontColor": solid(TEXT),
                        }
                    }
                ],
            },
            "drillFilterOtherVisuals": True,
        },
    }


# ---------- shared layout helpers ----------

def header(title, subtitle, z=1):
    """Big page header — 24px margin, full width, 86px high (matches reference)."""
    return textbox(
        "h", 24, 14, 1232, 86, z,
        [
            {"text": title, "size": "23pt", "bold": True, "color": TITLE},
            {"text": subtitle, "size": "11pt", "color": TEXT_DIM},
        ],
    )


def section_title(name, x, y, w, text, z):
    """Section title — 30px high, larger than the row labels we used to do."""
    return textbox(name, x, y, w, 30, z, [{"text": text, "size": "14pt", "bold": True, "color": TITLE}])


def small_label(name, x, y, w, text, z):
    """Used for KPI labels above cards — 26px high so 11pt fits comfortably."""
    return textbox(name, x, y, w, 26, z, [{"text": text, "size": "11pt", "bold": True, "color": SUBTITLE}])


# ---------- page 1: Resumen ----------

def build_page_resumen():
    visuals: list[dict] = []
    z = 0

    visuals.append(header(
        "ATAXX-ZERO — Catálogo de generaciones",
        "8 corridas de entrenamiento ejecutadas entre marzo y mayo 2026. Datos en runs_history/all_runs.csv.",
        z=1,
    )); z = 2

    # KPI row: 4 cards across 1232 width with 50px gaps
    kpi_y_label = 116
    kpi_y_card = 144
    card_w = 270
    card_h = 90
    gap = (1232 - 4 * card_w) // 3
    kpis = [
        ("Generaciones entrenadas", "Generaciones", ACCENT_DARK),
        ("Iteraciones máximas (1 run)", "Iteraciones", ACCENT_DARK),
        ("Mejor eval observada", "Mejor eval", ACCENT_RED),
        ("Score final vs sentinel", "vs sentinel", ACCENT_BLUE),
    ]
    for idx, (label_text, measure_name, color) in enumerate(kpis):
        x = 24 + idx * (card_w + gap)
        visuals.append(small_label(f"klbl{idx}", x, kpi_y_label, card_w, label_text, z)); z += 1
        visuals.append(card(f"kcard{idx}", x, kpi_y_card, card_w, card_h, z, measure=measure_name, value_color=color)); z += 1

    # Ranking table — title + explanatory text + table
    visuals.append(section_title("sec_rk", 24, 254, 1232, "Ranking de generaciones", z)); z += 1
    visuals.append(textbox(
        "rk_help", 24, 286, 1232, 32, z,
        [{
            "text": "Cada fila es una generación de modelo. Mejor eval = score máximo observado durante el entrenamiento (entre 0 = perdió todas y 1 = ganó todas). vs hard/apex/sentinel = score final contra cada heurística. Promedio heurístico = media de los tres niveles.",
            "size": "10pt",
            "color": TEXT_DIM,
        }],
    )); z += 1
    visuals.append(table_ex(
        "rk_table", 24, 322, 1232, 180, z,
        value_specs=[
            ("column", "codename"),
            ("column", "version"),
            ("measure", "Iter máx"),
            ("measure", "Mejor eval"),
            ("measure", "vs hard"),
            ("measure", "vs apex"),
            ("measure", "vs sentinel"),
            ("measure", "Promedio heurístico"),
        ],
        show_total=False,
    )); z += 1

    # Bar chart: eval peak per generación
    visuals.append(section_title("sec_bar", 24, 516, 1232, "Mejor eval observada por generación", z)); z += 1
    visuals.append(bar_chart(
        "bar_peak", 24, 552, 1232, 148, z,
        category=("column", "codename"),
        y_field=("measure", "Mejor eval"),
        color=ACCENT_DARK,
        sort_by=("measure", "Mejor eval", "Descending"),
    )); z += 1

    return {"name": "resumen", "displayName": "Resumen", "visuals": visuals}


# ---------- page 2: Curva de aprendizaje ----------

def build_page_curva():
    visuals: list[dict] = []
    z = 0

    visuals.append(header(
        "Curva de aprendizaje",
        "Cómo mejoró cada generación a lo largo del entrenamiento. Eje X = iteración (1 a 180). Eje Y = mejor eval observada en esa iteración (0 a 1).",
        z=1,
    )); z = 2

    # Slicer (left column)
    slicer_x = 24
    slicer_w = 270
    visuals.append(small_label("sl_lbl", slicer_x, 116, slicer_w, "Filtrar generación", z)); z += 1
    visuals.append(slicer(
        "sl_codename", slicer_x, 146, slicer_w, 380, z,
        prop="codename", header_text="Generación",
    )); z += 1

    # Reading guide under slicer
    visuals.append(textbox(
        "guide", slicer_x, 540, slicer_w, 160, z,
        [
            {"text": "Cómo leer:", "size": "11pt", "bold": True, "color": TITLE},
            {"text": "• líneas que suben a ~0.8 = aprendió", "size": "10pt", "color": TEXT_DIM},
            {"text": "• líneas pegadas al 0 = fracasó", "size": "10pt", "color": TEXT_DIM},
            {"text": "• inicios en -1 = sin eval aún", "size": "10pt", "color": TEXT_DIM},
        ],
    )); z += 1

    # Big line chart (right)
    line_x = slicer_x + slicer_w + 24
    line_w = 1280 - 24 - line_x
    visuals.append(section_title("ln_lbl", line_x, 116, line_w, "Mejor eval por iteración (todas las generaciones)", z)); z += 1
    visuals.append(line_chart(
        "ln_eval", line_x, 152, line_w, 534, z,
        category=("column", "iter"),
        y_field=("measure", "Mejor eval por iter"),
        legend=("column", "codename"),
        color=ACCENT_DARK,
    )); z += 1

    return {"name": "curva_aprendizaje", "displayName": "Curva de aprendizaje", "visuals": visuals}


# ---------- page 3: Entrenamiento ----------

def build_page_entrenamiento():
    """Página explicativa: 4 gráficos del training loop con título + descripción
    en lenguaje accesible. Layout 2x2 con bloques compactos para que no se solape
    nada con el auto-title interno del chart."""
    visuals: list[dict] = []
    z = 0

    # Header con explicación del aparente paradoja
    visuals.append(textbox(
        "h", 24, 14, 1232, 96, 1,
        [
            {"text": "Cómo aprendió el modelo (solo liga)", "size": "22pt", "bold": True, "color": TITLE},
            {
                "text": "Aviso importante: en estos 4 gráficos liga se ve cada vez PEOR durante el entrenamiento. Sin embargo en juego real es la mejor generación (0.81 vs heurísticas — ver pestaña Resumen). El motivo es que el modelo nunca juega solo: siempre usa MCTS como ayudante para elegir su movida. Estos números miden al modelo aislado, no su desempeño real.",
                "size": "10pt",
                "color": TEXT_DIM,
            },
        ],
    )); z = 2

    chart_w = (1232 - 24) // 2  # 604

    def chart_block(name_prefix, x, y_top, title_text, desc_text, measure_name, color):
        """Combined title+description textbox (50h) + chart (~210h)."""
        nonlocal z
        visuals.append(textbox(
            f"{name_prefix}_t", x, y_top, chart_w, 50, z,
            [
                {"text": title_text, "size": "13pt", "bold": True, "color": TITLE},
                {"text": desc_text, "size": "10pt", "color": TEXT_DIM},
            ],
        )); z += 1
        visuals.append(line_chart(
            f"{name_prefix}_ch", x, y_top + 56, chart_w, 210, z,
            category=("column", "iter"),
            y_field=("measure", measure_name),
            legend=("column", "codename"),
            color=color,
        )); z += 1

    top_y = 120
    chart_block(
        "tl", 24, top_y,
        "Error general del modelo",
        "Cuánto se equivocaba al predecir. Subir = empeora.",
        "Error general", ACCENT_DARK,
    )
    chart_block(
        "vl", 24 + chart_w + 24, top_y,
        "Error al predecir el ganador",
        "Adivina si va ganando o perdiendo. Subir = predicciones peores.",
        "Error al predecir ganador", ACCENT_RED,
    )

    bot_y = 398
    chart_block(
        "pl", 24, bot_y,
        "Error al sugerir jugada",
        "Propone qué movida hacer. Subir = sugerencias peores.",
        "Error al sugerir jugada", ACCENT_BLUE,
    )
    chart_block(
        "pa", 24 + chart_w + 24, bot_y,
        "Aciertos al sugerir jugada",
        "Coincide con la búsqueda MCTS. Bajar = se desincroniza.",
        "Aciertos al sugerir jugada", ACCENT_DARK,
    )

    return {"name": "entrenamiento", "displayName": "Entrenamiento", "visuals": visuals}


# ---------- writer ----------

def write_page(page_def):
    page_dir = PAGES_DIR / page_def["name"]
    if page_dir.exists():
        shutil.rmtree(page_dir)
    page_dir.mkdir(parents=True)
    page_json = {
        "$schema": PAGE_SCHEMA,
        "name": page_def["name"],
        "displayName": page_def["displayName"],
        "displayOption": "FitToPage",
        "height": PAGE_H,
        "width": PAGE_W,
    }
    (page_dir / "page.json").write_text(json.dumps(page_json, indent=2, ensure_ascii=False), encoding="utf-8")

    visuals_dir = page_dir / "visuals"
    visuals_dir.mkdir()
    for visual in page_def["visuals"]:
        v_dir = visuals_dir / visual["name"]
        v_dir.mkdir()
        (v_dir / "visual.json").write_text(json.dumps(visual, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    pages = [
        build_page_resumen(),
        build_page_curva(),
        build_page_entrenamiento(),
    ]

    for child in PAGES_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child)

    for page in pages:
        write_page(page)
        print(f"[ok] página: {page['displayName']}  ({len(page['visuals'])} visuales)")

    pages_meta = {
        "$schema": PAGES_META_SCHEMA,
        "pageOrder": [p["name"] for p in pages],
        "activePageName": pages[0]["name"],
    }
    (PAGES_DIR / "pages.json").write_text(json.dumps(pages_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(pages)} páginas escritas en {PAGES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
