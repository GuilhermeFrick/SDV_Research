"""
SOME/IP Packet Analyzer — Dash dashboard
Left panel : Wireshark-like packet list table
Right panel: SOME/IP header diagram filled with selected packet values
"""

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import os

from parser import parse_pcap, packets_to_rows

# ── colour palette ────────────────────────────────────────────────────────────
BG_PAGE   = "#0f1117"
BG_PANEL  = "#161b2e"
BG_TABLE  = "#1a2035"
BG_HEADER = "#1e2745"
ACCENT    = "#4e7fff"
TEXT      = "#d0d8f0"
SUBTLE    = "#6b7a9e"

FIELD_COLORS = {
    "service":   ("#151d38", "#7b9cff"),
    "method":    ("#1a1538", "#c07bff"),
    "length":    ("#1a1530", "#c07bff"),
    "client":    ("#141f20", "#7bffd9"),
    "session":   ("#141f20", "#7bffd9"),
    "proto_ver": ("#1a1f14", "#a8ff7b"),
    "iface_ver": ("#1a1f14", "#a8ff7b"),
    "msg_type":  ("#1e2818", "#d4ff7b"),
    "ret_code":  ("#1e2818", "#d4ff7b"),
    "payload":   ("#1a1010", "#ff9d7b"),
}


def field_box(label, value, key, byte_range, width="100%"):
    bg, fg = FIELD_COLORS[key]
    return html.Div([
        html.Div(byte_range, style={
            "fontSize": "9px", "color": SUBTLE, "marginBottom": "2px",
            "fontFamily": "monospace",
        }),
        html.Div(label, style={
            "fontSize": "11px", "color": fg, "fontWeight": "600",
            "marginBottom": "4px",
        }),
        html.Div(value, style={
            "fontSize": "14px", "color": fg, "fontFamily": "monospace",
            "fontWeight": "700", "wordBreak": "break-all",
        }),
    ], style={
        "background": bg,
        "border": f"1px solid {fg}44",
        "borderRadius": "6px",
        "padding": "8px 10px",
        "width": width,
        "boxSizing": "border-box",
    })


def build_header_diagram(row: dict | None):
    if row is None:
        placeholder = html.Div(
            "Selecione um pacote na tabela para visualizar o cabeçalho SOME/IP",
            style={"color": SUBTLE, "textAlign": "center", "marginTop": "80px",
                   "fontSize": "14px"},
        )
        return placeholder

    sid_str     = f"0x{row['_service_id']:04X}"
    mid_str     = f"0x{row['_method_id']:04X}"
    length_val  = row["Length"]
    client_val  = f"0x{row['_client_id']:04X}"
    session_val = f"0x{row['_session_id']:04X}"
    pv_val      = f"0x{row['_proto_ver']:02X}"
    iv_val      = f"0x{row['_iface_ver']:02X}"
    mt_val      = f"0x{row['_msg_type']:02X}  {row['Msg Type']}"
    rc_val      = f"0x{row['_return_code']:02X}  {row['_rc_name']}"
    pay_hex     = row["_payload_hex"] or "—"
    pay_display = (pay_hex[:48] + "…") if len(pay_hex) > 48 else pay_hex

    row1 = html.Div([
        field_box("Service ID", sid_str, "service", "Bytes 0–1", "50%"),
        field_box("Method ID",  mid_str, "method",  "Bytes 2–3", "50%"),
    ], style={"display": "flex", "gap": "8px", "marginBottom": "8px"})

    row2 = html.Div([
        field_box("Length (after byte 7)", str(length_val), "length", "Bytes 4–7", "100%"),
    ], style={"marginBottom": "8px"})

    row3 = html.Div([
        field_box("Client ID",  client_val,  "client",  "Bytes 8–9",  "50%"),
        field_box("Session ID", session_val, "session", "Bytes 10–11", "50%"),
    ], style={"display": "flex", "gap": "8px", "marginBottom": "8px"})

    row4 = html.Div([
        field_box("Protocol Ver", pv_val, "proto_ver", "Byte 12", "25%"),
        field_box("Interface Ver", iv_val, "iface_ver", "Byte 13", "25%"),
        field_box("Msg Type",    mt_val, "msg_type", "Byte 14", "25%"),
        field_box("Return Code", rc_val, "ret_code", "Byte 15", "25%"),
    ], style={"display": "flex", "gap": "8px", "marginBottom": "8px"})

    row5 = html.Div([
        field_box("Payload (hex, first 32 bytes)", pay_display, "payload", f"Bytes 16+ ({row['Pay Len']} bytes total)", "100%"),
    ], style={"marginBottom": "0"})

    summary = html.Div([
        html.Span(f"#{row['#']}  ", style={"color": ACCENT, "fontWeight": "700"}),
        html.Span(f"{row['Source']} → {row['Destination']}  ", style={"color": TEXT}),
        html.Span(f"[{row['Proto']}]", style={"color": SUBTLE}),
    ], style={"marginBottom": "16px", "fontSize": "13px", "fontFamily": "monospace"})

    return html.Div([summary, row1, row2, row3, row4, row5])


# ── App layout ────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="SOME/IP Analyzer")
app.layout = html.Div([
    # title bar
    html.Div([
        html.Span("SOME/IP Packet Analyzer", style={
            "fontSize": "18px", "fontWeight": "700", "color": TEXT,
        }),
        html.Span(" — SDV Research", style={"color": SUBTLE, "fontSize": "14px"}),
    ], style={
        "background": BG_HEADER, "padding": "12px 20px",
        "borderBottom": f"1px solid {ACCENT}44",
    }),

    # file input bar
    html.Div([
        html.Label("PCAP file path:", style={"color": SUBTLE, "marginRight": "10px", "fontSize": "13px"}),
        dcc.Input(
            id="pcap-path",
            type="text",
            placeholder="C:\\path\\to\\file.pcap",
            debounce=False,
            style={
                "width": "520px", "background": BG_TABLE, "color": TEXT,
                "border": f"1px solid {ACCENT}55", "borderRadius": "5px",
                "padding": "6px 10px", "fontFamily": "monospace", "fontSize": "13px",
            },
        ),
        html.Button("Load", id="load-btn", n_clicks=0, style={
            "marginLeft": "10px", "background": ACCENT, "color": "#fff",
            "border": "none", "borderRadius": "5px", "padding": "6px 18px",
            "cursor": "pointer", "fontWeight": "600",
        }),
        html.Span(id="load-status", style={"marginLeft": "14px", "color": SUBTLE, "fontSize": "12px"}),
    ], style={"padding": "10px 20px", "display": "flex", "alignItems": "center",
              "background": BG_PANEL, "borderBottom": f"1px solid #2a3550"}),

    # main area
    html.Div([
        # LEFT — packet table
        html.Div([
            html.Div("Packet List", style={
                "color": SUBTLE, "fontSize": "11px", "fontWeight": "600",
                "textTransform": "uppercase", "letterSpacing": "1px",
                "marginBottom": "8px",
            }),
            dash_table.DataTable(
                id="packet-table",
                columns=[{"name": c, "id": c} for c in [
                    "#", "Time", "Source", "Destination", "Proto",
                    "Service ID", "Method ID", "Msg Type", "Length", "Pay Len",
                ]],
                data=[],
                page_size=300,
                fixed_rows={"headers": True},
                style_table={"height": "calc(100vh - 180px)", "overflowY": "auto"},
                style_cell={
                    "background": BG_TABLE, "color": TEXT,
                    "fontFamily": "monospace", "fontSize": "12px",
                    "border": f"1px solid #2a3550",
                    "padding": "4px 8px", "textAlign": "left",
                    "whiteSpace": "nowrap", "overflow": "hidden",
                    "textOverflow": "ellipsis", "maxWidth": "160px",
                },
                style_header={
                    "background": BG_HEADER, "color": ACCENT,
                    "fontWeight": "700", "fontSize": "11px",
                    "border": f"1px solid #2a3550",
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "background": "#1c2340"},
                    {"if": {"filter_query": '{Proto} = "UDP"'},
                     "color": "#7bffd9"},
                    {"if": {"filter_query": '{Service ID} = "0xFFFF"'},
                     "color": "#ffd97b"},
                ],
                row_selectable="single",
                selected_rows=[],
            ),
        ], style={
            "width": "55%", "padding": "14px", "background": BG_PANEL,
            "borderRight": f"1px solid #2a3550",
        }),

        # RIGHT — header diagram
        html.Div([
            html.Div("SOME/IP Header", style={
                "color": SUBTLE, "fontSize": "11px", "fontWeight": "600",
                "textTransform": "uppercase", "letterSpacing": "1px",
                "marginBottom": "14px",
            }),
            html.Div(
                build_header_diagram(None),
                id="header-diagram",
                style={"overflowY": "auto", "maxHeight": "calc(100vh - 200px)"},
            ),
        ], style={
            "width": "45%", "padding": "14px", "background": BG_PANEL,
        }),
    ], style={"display": "flex", "height": "calc(100vh - 88px)", "overflow": "hidden"}),

    # hidden store for full row data
    dcc.Store(id="packet-store", data=[]),
], style={"background": BG_PAGE, "minHeight": "100vh", "fontFamily": "sans-serif"})


# ── Callbacks ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("packet-table", "data"),
    Output("packet-store", "data"),
    Output("load-status", "children"),
    Output("packet-table", "selected_rows"),
    Input("load-btn", "n_clicks"),
    State("pcap-path", "value"),
    prevent_initial_call=True,
)
def load_pcap(n, path):
    if not path or not path.strip():
        return [], [], "Insira o caminho do arquivo PCAP.", []
    path = path.strip().strip('"').strip("'")
    if not os.path.isfile(path):
        return [], [], f"Arquivo não encontrado: {path}", []
    try:
        packets = parse_pcap(path, max_packets=20_000)
        if not packets:
            return [], [], "Nenhum pacote SOME/IP encontrado.", []
        rows = packets_to_rows(packets)
        # strip hidden fields for the visible table, keep everything in store
        visible_cols = ["#", "Time", "Source", "Destination", "Proto",
                        "Service ID", "Method ID", "Msg Type", "Length", "Pay Len"]
        table_rows = [{k: r[k] for k in visible_cols} for r in rows]
        status = f"{len(packets):,} pacotes carregados — {os.path.basename(path)}"
        return table_rows, rows, status, []
    except Exception as e:
        return [], [], f"Erro: {e}", []


@app.callback(
    Output("header-diagram", "children"),
    Input("packet-table", "selected_rows"),
    State("packet-store", "data"),
)
def update_diagram(selected, store):
    if not selected or not store:
        return build_header_diagram(None)
    idx = selected[0]
    if idx >= len(store):
        return build_header_diagram(None)
    row = store[idx]
    return build_header_diagram(row)


if __name__ == "__main__":
    app.run(debug=True, port=8050)
