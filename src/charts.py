"""
Génération des graphiques.

Le graphique vient AVANT le texte : il sert de preuve. Le commerçant voit la
courbe qui descend, puis lit pourquoi et quoi faire. Sans le visuel d'abord,
l'interprétation ressemble à une affirmation gratuite.

Les figures sont construites depuis `facts.json` uniquement — jamais depuis
le DataFrame. Ce qui est tracé est donc exactement ce qui est interprété.
"""

from __future__ import annotations

import plotly.graph_objects as go

# Palette sobre, lisible en projection comme à l'écran
BLEU, ROUGE, VERT, GRIS = "#2563eb", "#dc2626", "#059669", "#94a3b8"
SERIES = ["#2563eb", "#059669", "#f59e0b", "#8b5cf6", "#ec4899",
          "#06b6d4", "#dc2626", "#84cc16"]

MISE_EN_PAGE = dict(
    template="plotly_white",
    font=dict(family="system-ui, -apple-system, sans-serif", size=13),
    margin=dict(l=60, r=30, t=60, b=60),
    height=380,
    hovermode="x unified",
    showlegend=False,
)


def _format_valeur(v: float, unite: str) -> str:
    if abs(v) >= 1_000_000:
        t = f"{v / 1_000_000:.1f} M"
    elif abs(v) >= 1000:
        t = f"{v / 1000:.1f} k"
    else:
        t = f"{v:,.0f}".replace(",", " ")
    return f"{t} {unite}".strip()


def _multi_series(facts: dict) -> bool:
    return any("serie" in p for p in facts["data_points"])


def build_figure(facts: dict) -> go.Figure:
    """Construit la figure correspondant à un bloc de facts."""
    points = facts["data_points"]
    chart = facts.get("chart", "bar")
    unite = facts.get("unit", "")
    stats = facts.get("computed_stats", {})

    if _multi_series(facts):
        fig = _figure_multi(points, chart)
    elif chart in {"line", "area"}:
        fig = _figure_ligne(points, facts, unite, chart == "area")
    else:
        fig = _figure_barres(points, facts, unite)

    fig.update_layout(
        title=dict(text=facts["title"], font=dict(size=16), x=0, xanchor="left"),
        **MISE_EN_PAGE,
    )
    fig.update_yaxes(title_text=facts.get("measure_label", ""),
                     gridcolor="#f1f5f9", zerolinecolor="#e2e8f0")
    fig.update_xaxes(title_text=(facts["dimension_labels"][0]
                                 if facts.get("dimension_labels") else ""),
                     gridcolor="#f8fafc")

    # La dernière période partielle est tracée mais signalée : la masquer
    # ferait disparaître une information réelle, la laisser sans mention
    # laisserait croire à un effondrement.
    if stats.get("last_period_incomplete") and points:
        fig.add_annotation(
            x=points[-1]["dimension"], y=points[-1]["value"],
            text="période incomplète", showarrow=True, arrowhead=0,
            ax=0, ay=-32, font=dict(size=11, color=GRIS),
            bgcolor="rgba(255,255,255,0.85)",
        )
    return fig


def _figure_ligne(points, facts, unite, aire: bool) -> go.Figure:
    x = [p["dimension"] for p in points]
    y = [p["value"] for p in points]
    stats = facts.get("computed_stats", {})

    # Couleur porteuse de sens : rouge si la tendance baisse
    baisse = any(p["pattern_id"] in {"trend_decline", "basket_erosion"}
                 for p in facts.get("detected_patterns", []))
    couleur = ROUGE if baisse else BLEU

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers", line=dict(color=couleur, width=2.5),
        marker=dict(size=6), fill="tozeroy" if aire else None,
        fillcolor=f"rgba(37,99,235,0.10)" if aire else None,
        hovertemplate="%{x}<br><b>%{y:,.0f} " + unite + "</b><extra></extra>",
    ))

    # Droite de tendance, uniquement si l'ajustement est significatif
    if stats.get("trend_r2", 0) > 0.25 and stats.get("trend_slope") is not None:
        n = len(y) - (1 if stats.get("last_period_incomplete") else 0)
        if n >= 3:
            pente = stats["trend_slope"]
            depart = stats.get("first_value", y[0])
            fig.add_trace(go.Scatter(
                x=x[:n], y=[depart + pente * i for i in range(n)],
                mode="lines", line=dict(color=GRIS, width=1.5, dash="dot"),
                hoverinfo="skip",
            ))
    return fig


def _figure_barres(points, facts, unite) -> go.Figure:
    x = [p["dimension"] for p in points]
    y = [p["value"] for p in points]

    # Mettre en évidence ce qui compte : la barre dominante en cas de
    # concentration, le pic en cas de rythme hebdomadaire.
    ids = {p["pattern_id"] for p in facts.get("detected_patterns", [])}
    couleurs = [BLEU] * len(y)
    if ids & {"pareto_concentration", "geographic_concentration", "day_of_week_peak"} and y:
        imax = y.index(max(y))
        couleurs = [GRIS if i != imax else VERT for i in range(len(y))]

    fig = go.Figure(go.Bar(
        x=x, y=y, marker_color=couleurs,
        text=[_format_valeur(v, unite) for v in y],
        textposition="outside", textfont=dict(size=11),
        hovertemplate="%{x}<br><b>%{y:,.0f} " + unite + "</b><extra></extra>",
    ))
    fig.update_yaxes(rangemode="tozero")
    return fig


def _figure_multi(points, chart) -> go.Figure:
    series = {}
    for p in points:
        series.setdefault(p.get("serie", ""), []).append((p["dimension"], p["value"]))

    fig = go.Figure()
    for i, (nom, vals) in enumerate(series.items()):
        fig.add_trace(go.Bar(
            name=nom, x=[v[0] for v in vals], y=[v[1] for v in vals],
            marker_color=SERIES[i % len(SERIES)],
            hovertemplate="%{x}<br>" + nom + " : <b>%{y:,.0f}</b><extra></extra>",
        ))
    fig.update_layout(barmode="stack" if chart == "stacked_bar" else "group")
    fig.update_layout(showlegend=True,
                      legend=dict(orientation="h", y=-0.2, x=0))
    return fig


def figure_to_html(fig: go.Figure, include_js: bool = False) -> str:
    """Exporte en HTML autonome, pour l'intégration dans le rapport."""
    return fig.to_html(full_html=False,
                       include_plotlyjs="cdn" if include_js else False,
                       config={"displayModeBar": False})
