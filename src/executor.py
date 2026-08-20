"""
Exécution des spécifications et détection des patterns métier.

Étape [6b] du pipeline. Produit `facts.json` — LE SEUL CONTENU que verra
l'interprète.

RÈGLE ABSOLUE
    Tout nombre qui apparaîtra dans le texte final doit exister dans ce fichier.
    Le vérificateur (etape 8) l'impose mécaniquement.

Les patterns métier sont détectés ICI, par du code, jamais par le modèle.
Le LLM ne fera que les formuler dans le contexte du commerçant.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

GRAIN_FR = {
    "day": "jour", "week": "semaine", "month": "mois", "quarter": "trimestre",
    "year": "année", "dayofweek": "jour de la semaine", "hour": "heure",
}

JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


# --------------------------------------------------------------------------
# Exécution
# --------------------------------------------------------------------------

def _apply_filters(df: pd.DataFrame, filters: list) -> tuple[pd.DataFrame, list]:
    appliques = []
    for f in filters or []:
        col, op, val = f.get("column"), f.get("op"), f.get("value")
        if col not in df.columns:
            continue
        s = df[col]
        try:
            if op == "eq":
                m = s.astype(str) == str(val)
            elif op == "neq":
                m = s.astype(str) != str(val)
            elif op == "in":
                m = s.astype(str).isin([str(v) for v in val])
            elif op == "not_in":
                m = ~s.astype(str).isin([str(v) for v in val])
            elif op == "gt":
                m = pd.to_numeric(s, errors="coerce") > val
            elif op == "gte":
                m = pd.to_numeric(s, errors="coerce") >= val
            elif op == "lt":
                m = pd.to_numeric(s, errors="coerce") < val
            elif op == "lte":
                m = pd.to_numeric(s, errors="coerce") <= val
            elif op == "between":
                v = pd.to_numeric(s, errors="coerce")
                m = (v >= val[0]) & (v <= val[1])
            elif op == "not_null":
                m = s.notna()
            else:
                continue
        except Exception:
            continue
        df = df[m]
        appliques.append(f)
    return df, appliques


def _temporal_key(s: pd.Series, grain: str, dayfirst: bool = True) -> pd.Series:
    """
    Convertit une date en clé d'agrégation lisible.

    `dayfirst=True` est INDISPENSABLE. Sans lui, pandas applique la convention
    américaine : « 04/10/2023 » devient le 10 avril au lieu du 4 octobre. Sur
    un fichier réel, cela étalait 25 mois de ventes sur 36 mois — avec des mois
    de bord presque vides — et produisait une « hausse de 983,5 % » entièrement
    fabriquée.

    Le profileur avait pourtant identifié le bon format. Le défaut venait de ce
    reparsing en aval, qui ignorait ce qui avait déjà été établi.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        d = s
    else:
        from .profiler import detecter_ordre_jour_mois
        ordre, conf = detecter_ordre_jour_mois(s.dropna())
        if conf >= 0.7:
            dayfirst = (ordre == "dayfirst")
        d = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=dayfirst)
    if grain == "day":
        return d.dt.strftime("%Y-%m-%d")
    if grain == "week":
        return d.dt.to_period("W").astype(str).str.slice(0, 10)
    if grain == "quarter":
        return d.dt.to_period("Q").astype(str)
    if grain == "year":
        return d.dt.year.astype("Int64").astype(str)
    if grain == "dayofweek":
        return d.dt.dayofweek.map(dict(enumerate(JOURS)))
    if grain == "hour":
        return d.dt.hour.astype("Int64").astype(str).str.zfill(2) + "h"
    return d.dt.to_period("M").astype(str)          # month par défaut


def execute_spec(spec: dict, df: pd.DataFrame) -> dict | None:
    """
    Exécute une spécification. Retourne les données agrégées, ou None si le
    résultat est inexploitable (V9 : moins de 2 lignes).
    """
    mesure = spec["measure"]
    col_m, agg = mesure["column"], mesure["agg"]
    dims = spec["dimensions"]

    data, filtres = _apply_filters(df, spec.get("filters"))
    if data.empty:
        return None

    # Clés de regroupement
    cles, labels = [], []
    for d in dims:
        col = d["column"]
        if d["type"] == "temporal":
            grain = d.get("grain", "month")
            k = _temporal_key(data[col], grain)
            labels.append(f"{col} ({GRAIN_FR.get(grain, grain)})")
        else:
            k = data[col].astype(str)
            labels.append(col)
        k.name = f"__dim{len(cles)}"
        cles.append(k)

    # Une ligne sans valeur de dimension n'est pas imputée : elle est exclue,
    # et le compte des lignes écartées est conservé pour le rapport.
    valides = pd.concat(cles, axis=1).notna().all(axis=1)
    n_exclues = int((~valides).sum())
    data, cles = data[valides], [k[valides] for k in cles]
    if data.empty:
        return None

    # Effectif par groupe : indispensable pour repérer une période partielle
    # sur une MOYENNE. Une moyenne ne s'effondre pas quand le mois est
    # incomplet — seul le nombre de lignes le révèle.
    effectifs = data.groupby(cles, dropna=True, observed=True)[col_m].size()

    grouped = data.groupby(cles, dropna=True, observed=True)[col_m]
    fonctions = {
        "sum": "sum", "mean": "mean", "median": "median", "count": "count",
        "min": "min", "max": "max", "std": "std", "count_distinct": "nunique",
    }
    res = grouped.agg(fonctions[agg]).reset_index()
    res.columns = [f"dim{i}" for i in range(len(cles))] + ["value"]
    res = res.dropna(subset=["value"])
    if len(res) < 2:
        return None

    # Tri et limite
    tri = spec.get("sort") or {}
    par = "value" if tri.get("by") == "measure" else "dim0"
    croissant = tri.get("order", "asc") == "asc"
    temporel = dims[0]["type"] == "temporal"

    if par == "dim0" and temporel:
        res = res.sort_values("dim0", ascending=True)
    else:
        res = res.sort_values(par, ascending=croissant)

    total_avant = float(res["value"].sum())
    n_avant = len(res)
    limite = spec.get("limit")
    if limite and len(res) > limite:
        if not temporel:
            res = res.nlargest(limite, "value") if not croissant else res.nsmallest(limite, "value")
        else:
            res = res.tail(limite)

    # Effectifs alignés sur les groupes retenus
    res = res.reset_index(drop=True)
    try:
        cle_eff = effectifs.reset_index()
        cle_eff.columns = [f"dim{i}" for i in range(len(cles))] + ["n_lignes"]
        res = res.merge(cle_eff, on=[f"dim{i}" for i in range(len(cles))], how="left")
    except Exception:
        res["n_lignes"] = np.nan

    return {
        "table": res,
        "n_groupes_total": n_avant,
        "n_groupes_affiches": len(res),
        "total": total_avant,
        "n_lignes_utilisees": int(valides.sum()),
        "n_lignes_exclues": n_exclues,
        "filtres_appliques": filtres,
        "dim_labels": labels,
        "temporel": temporel,
    }


# --------------------------------------------------------------------------
# Statistiques
# --------------------------------------------------------------------------

def compute_stats(res: dict) -> dict:
    t = res["table"]
    v = t["value"].astype(float)
    st = {
        "n_points": len(t),
        "total": round(float(v.sum()), 2),
        "mean": round(float(v.mean()), 2),
        "median": round(float(v.median()), 2),
        "max": {"dimension": str(t.loc[v.idxmax(), "dim0"]), "value": round(float(v.max()), 2)},
        "min": {"dimension": str(t.loc[v.idxmin(), "dim0"]), "value": round(float(v.min()), 2)},
    }

    if res["temporel"] and len(v) >= 3:
        # La dernière période est souvent partielle : un fichier exporté le 4
        # du mois contient 4 jours de ventes, pas un mois. Sans correction, le
        # système annonçait « −85 % » — un chiffre faux qui affolerait le
        # commerçant. On détecte le cas et on calcule la tendance SANS cette
        # période, tout en la conservant dans le graphique.
        # Les périodes de BORD peuvent être partielles des DEUX côtés : un
        # export du 4 février a un dernier mois tronqué, un export commencé
        # le 31 mars a un PREMIER mois d'une seule journée. Ne contrôler que
        # la dernière laissait passer « +22 461 % » sur un fichier où
        # l'activité baissait en réalité de 18,5 %.
        t = res["table"]
        eff = (t["n_lignes"].astype(float)
               if "n_lignes" in t.columns and t["n_lignes"].notna().all() else None)
        debut, fin = 0, len(v)
        exclues = []

        if len(v) >= 4:
            centre = v.iloc[1:-1] if len(v) > 3 else v
            med_val = float(centre.median())
            med_eff = float(eff.iloc[1:-1].median()) if eff is not None else None

            def partielle(i):
                if med_val and v.iloc[i] < med_val * 0.5:
                    return True
                if med_eff and eff is not None and eff.iloc[i] < med_eff * 0.5:
                    return True
                return False

            if partielle(0):
                debut = 1
                exclues.append({"periode": str(t["dim0"].iloc[0]),
                                "position": "début",
                                "valeur": round(float(v.iloc[0]), 2),
                                "n_lignes": int(eff.iloc[0]) if eff is not None else None})
            if partielle(len(v) - 1):
                fin = len(v) - 1
                exclues.append({"periode": str(t["dim0"].iloc[-1]),
                                "position": "fin",
                                "valeur": round(float(v.iloc[-1]), 2),
                                "n_lignes": int(eff.iloc[-1]) if eff is not None else None})

        v_calc = v.iloc[debut:fin] if fin - debut >= 2 else v
        if exclues:
            st["periodes_incompletes"] = exclues
            st["last_period_incomplete"] = True      # compatibilité
            st["last_period_excluded"] = ", ".join(e["periode"] for e in exclues)
            st["last_period_value"] = exclues[-1]["valeur"]

        st["first_value"] = round(float(v_calc.iloc[0]), 2)
        st["last_value"] = round(float(v_calc.iloc[-1]), 2)
        st["first_period"] = str(t["dim0"].iloc[debut])
        st["last_period"] = str(t["dim0"].iloc[debut + len(v_calc) - 1])
        st["change_absolute"] = round(float(v_calc.iloc[-1] - v_calc.iloc[0]), 2)
        if v_calc.iloc[0]:
            st["change_pct"] = round(float((v_calc.iloc[-1] / v_calc.iloc[0] - 1) * 100), 1)

        # Régression linéaire simple : pente et qualité d'ajustement
        x = np.arange(len(v_calc), dtype=float)
        if len(v_calc) >= 3:
            pente, ordonnee = np.polyfit(x, v_calc.values, 1)
            st["trend_slope"] = round(float(pente), 3)
            pred = pente * x + ordonnee
            ss_res = float(((v_calc.values - pred) ** 2).sum())
            ss_tot = float(((v_calc.values - v_calc.mean()) ** 2).sum())
            r2 = round(1 - ss_res / ss_tot, 3) if ss_tot else 0.0
            st["trend_r2"] = r2

            # QUALIFICATION DE LA TENDANCE.
            #
            # Comparer le premier et le dernier point d'une série est très
            # sensible au bruit : sur un fichier réel, cela donnait « −14,9 % »
            # alors que la pente était POSITIVE et le r² de 0,005. Le chiffre
            # était exact, sa lecture était fausse.
            #
            # `trend_direction` dit ce qu'on a le droit d'affirmer, et
            # `volatilite_pct` donne l'amplitude normale des variations.
            moy = float(v_calc.mean()) or 1.0
            volatilite = float(v_calc.std()) / abs(moy) * 100
            st["volatilite_pct"] = round(volatilite, 1)

            pente_relative = abs(pente) / abs(moy) * 100
            variation = st.get("change_pct", 0.0)

            # La pente et la variation début→fin doivent CONCORDER. Sur une
            # série saisonnière, une pente positive peut coexister avec des
            # points de bord en baisse : annoncer « une hausse de −22,4 % »
            # n'a aucun sens pour un lecteur. En cas de divergence, on ne
            # tranche pas.
            concordent = (pente > 0) == (variation > 0) if variation else True

            if r2 < 0.25 or pente_relative < 1.0 or not concordent:
                st["trend_direction"] = "stable"
                if not concordent:
                    st["trend_comment"] = (
                        f"Pas de direction claire : le niveau global progresse, "
                        f"mais le début et la fin de la période ne le reflètent "
                        f"pas. Les variations d'une période à l'autre atteignent "
                        f"{volatilite:.1f} %.")
                else:
                    st["trend_comment"] = (
                        f"Aucune tendance nette : les variations d'une période à "
                        f"l'autre ({volatilite:.1f} %) sont du même ordre que "
                        f"l'écart entre le début et la fin.")
            elif pente > 0:
                st["trend_direction"] = "hausse"
            else:
                st["trend_direction"] = "baisse"

    if not res["temporel"] and len(v) >= 3:
        tri = v.sort_values(ascending=False)
        cumul = tri.cumsum() / tri.sum()
        top20 = max(1, int(len(tri) * 0.2))
        st["top20_share_pct"] = round(float(cumul.iloc[top20 - 1] * 100), 1)
        st["top1_share_pct"] = round(float(tri.iloc[0] / tri.sum() * 100), 1)
        st["n_pour_80pct"] = int((cumul <= 0.8).sum() + 1)

    return st


# --------------------------------------------------------------------------
# Patterns métier — détectés par le code, jamais par le modèle
# --------------------------------------------------------------------------

def detect_patterns(res: dict, stats: dict, spec: dict, contexte: dict) -> list[dict]:
    """
    Chaque pattern associe une situation détectable à ses causes probables
    et aux actions recommandées. C'est ce qui transforme « le LLM raconte des
    banalités » en « le système diagnostique et recommande ».
    """
    out = []
    t = res["table"]
    v = t["value"].astype(float)
    role = contexte.get("business_role")

    # --- TENDANCE ---
    if res["temporel"] and stats.get("trend_slope") is not None and len(v) >= 4:
        pente, r2 = stats["trend_slope"], stats.get("trend_r2", 0)
        variation = stats.get("change_pct", 0)
        moyenne = stats["mean"] or 1
        direction = stats.get("trend_direction", "stable")

        # Une série qualifiée « stable » ne déclenche AUCUN pattern de tendance,
        # même si l'écart entre premier et dernier point paraît important.
        if direction == "baisse" and r2 > 0.25 and abs(pente) > moyenne * 0.02:
            out.append({
                "pattern_id": "trend_decline", "confidence": round(min(0.95, r2 + 0.3), 2),
                "evidence": {"variation_pct": variation, "pente": pente, "r2": r2},
                "causes": ["concurrence accrue", "saisonnalité", "rupture de stock",
                           "baisse du trafic"],
                "actions": ["Comparer vos prix à ceux de vos concurrents",
                            "Vérifier vos stocks sur la période concernée",
                            "Relancer les clients qui n'ont pas commandé depuis 3 mois"],
            })
        elif direction == "hausse" and r2 > 0.25 and pente > moyenne * 0.02:
            out.append({
                "pattern_id": "trend_growth", "confidence": round(min(0.95, r2 + 0.3), 2),
                "evidence": {"variation_pct": variation, "pente": pente, "r2": r2},
                "causes": ["acquisition efficace", "saisonnalité favorable",
                           "élargissement de l'offre"],
                "actions": ["Identifier ce qui a changé sur la période et le reproduire",
                            "Anticiper vos stocks pour soutenir la croissance",
                            "Renforcer le budget sur le canal qui fonctionne"],
            })

        # Pic isolé — uniquement sur un axe CHRONOLOGIQUE.
        # Sur « jour de la semaine » ou « heure », un pic n'est pas une anomalie
        # ponctuelle mais un rythme régulier : c'est day_of_week_peak qui
        # s'applique, avec des recommandations toutes différentes.
        grain_actuel = (spec["dimensions"][0].get("grain") if spec["dimensions"] else None)
        cyclique = grain_actuel in {"dayofweek", "hour"}
        med = float(v.median())
        mad = float((v - med).abs().median())
        if mad > 0 and not cyclique:
            z = 0.6745 * (v - med).abs() / mad
            pics = t[(z > 3.5) & (v > med)]
            if len(pics) and len(pics) <= max(2, len(v) // 5):
                out.append({
                    "pattern_id": "anomaly_spike", "confidence": 0.8,
                    "evidence": {"periodes": [str(x) for x in pics["dim0"].tolist()[:3]],
                                 "valeur_max": round(float(pics["value"].max()), 2),
                                 "niveau_habituel": round(med, 2)},
                    "causes": ["opération commerciale ponctuelle", "relais médiatique",
                              "commande exceptionnelle"],
                    "actions": ["Identifier ce qui s'est passé à cette période",
                                "Reproduire l'opération si elle est reproductible"],
                })

    # --- CONCENTRATION ---
    if not res["temporel"] and len(v) >= 4:
        part20 = stats.get("top20_share_pct", 0)
        part1 = stats.get("top1_share_pct", 0)

        if part20 >= 75:
            top = t.nlargest(1, "value").iloc[0]
            out.append({
                "pattern_id": "pareto_concentration", "confidence": 0.9,
                "evidence": {"part_top20_pct": part20,
                             "n_pour_80pct": stats.get("n_pour_80pct"),
                             "premier": str(top["dim0"]),
                             "part_premier_pct": part1},
                "causes": ["dépendance à quelques produits ou zones"],
                "actions": ["Sécuriser l'approvisionnement de vos meilleures ventes",
                            "Tester des produits complémentaires pour réduire la dépendance",
                            "Concentrer votre budget publicitaire sur ce qui rapporte"],
            })

    # Concentration géographique : traitée à part car un commerce n'a souvent
    # que 2 ou 3 zones — le seuil de 4 modalités du bloc précédent l'excluait.
    if not res["temporel"] and len(v) >= 2 and role == "region":
        part1 = stats.get("top1_share_pct") or round(float(v.max() / v.sum() * 100), 1)
        if part1 >= 50:
            out.append({
                "pattern_id": "geographic_concentration", "confidence": 0.85,
                "evidence": {"zone": str(t.loc[v.idxmax(), "dim0"]), "part_pct": part1,
                             "n_zones": len(v)},
                "causes": ["zone de chalandise étroite", "frais de port dissuasifs ailleurs"],
                "actions": ["Tester la publicité sur les zones voisines",
                            "Vérifier que vos frais de livraison ne bloquent pas ailleurs"],
            })

    if not res["temporel"] and len(v) >= 4:
        # Queue morte : beaucoup de modalités pour presque rien
        tri = v.sort_values(ascending=False)
        part = tri / tri.sum()
        faibles = int((part < 0.01).sum())
        if faibles >= max(3, len(tri) * 0.3):
            out.append({
                "pattern_id": "long_tail_dead_stock", "confidence": 0.75,
                "evidence": {"n_faibles": faibles, "n_total": len(tri),
                             "part_cumulee_pct": round(float(part[part < 0.01].sum() * 100), 1)},
                "causes": ["catalogue trop large", "produits mal référencés"],
                "actions": ["Déstocker ou retirer les références qui ne se vendent pas",
                            "Libérer la trésorerie immobilisée dans ces stocks"],
            })

    # --- JOUR DE LA SEMAINE ---
    grain = (spec["dimensions"][0].get("grain") if spec["dimensions"] else None)
    if grain == "dayofweek" and len(v) >= 5:
        moy = float(v.mean())
        pics = t[v >= moy * 1.4]
        if len(pics):
            creux = t.loc[v.idxmin()]
            out.append({
                "pattern_id": "day_of_week_peak", "confidence": 0.85,
                "evidence": {"jours_forts": [str(x) for x in pics["dim0"].tolist()],
                             "part_max_pct": round(float(v.max() / v.sum() * 100), 1),
                             "jour_faible": str(creux["dim0"])},
                "causes": ["habitude d'achat non exploitée"],
                "actions": ["Lancer vos campagnes la veille des jours forts",
                            "Renforcer vos stocks avant ces journées",
                            "Tester une promotion sur les jours creux"],
            })

    # --- SAISONNALITÉ ---
    if grain == "month" and len(v) >= 8:
        med = float(v.median())
        pics = t[v >= med * 1.8]
        if len(pics) and len(pics) <= len(v) // 3:
            out.append({
                "pattern_id": "seasonal_peak", "confidence": 0.8,
                "evidence": {"periodes_fortes": [str(x) for x in pics["dim0"].tolist()[:4]],
                             "niveau_median": round(med, 2)},
                "causes": ["saisonnalité du métier", "fêtes ou événements"],
                "actions": ["Constituer vos stocks 6 semaines avant ces périodes",
                            "Concentrer votre budget publicitaire à l'approche du pic"],
            })

    # --- PANIER MOYEN ---
    if (spec["measure"]["agg"] == "mean" and role == "revenue"
            and res["temporel"] and stats.get("change_pct", 0) < -10
            and stats.get("trend_direction") == "baisse"):
        out.append({
            "pattern_id": "basket_erosion", "confidence": 0.8,
            "evidence": {"variation_pct": stats["change_pct"],
                         "debut": stats.get("first_value"), "fin": stats.get("last_value")},
            "causes": ["promotions trop agressives", "mix produit dégradé",
                       "montée des petites commandes"],
            "actions": ["Créer des lots ou des offres groupées",
                        "Fixer un seuil de livraison gratuite au-dessus du panier actuel",
                        "Remonter vos produits à forte valeur sur la page d'accueil"],
        })

    # --- RÉTENTION ---
    if (spec["measure"]["agg"] == "count_distinct" and role == "customer_id"
            and res["temporel"]):
        n_clients = contexte.get("n_clients_uniques")
        n_cmd = contexte.get("n_commandes")
        if n_clients and n_cmd:
            cmd_par_client = n_cmd / n_clients
            if cmd_par_client < 1.3:
                out.append({
                    "pattern_id": "single_purchase_dominance", "confidence": 0.85,
                    "evidence": {"commandes_par_client": round(cmd_par_client, 2),
                                 "n_clients": n_clients, "n_commandes": n_cmd},
                    "causes": ["produit à achat unique", "aucune relance après achat",
                               "acquisition sans fidélisation"],
                    "actions": ["Mettre en place une relance par e-mail 15 jours après l'achat",
                                "Offrir une réduction sur la deuxième commande",
                                "Comparer le coût d'acquisition au revenu par client"],
                })
    return out


# --------------------------------------------------------------------------
# facts.json
# --------------------------------------------------------------------------

def build_facts(spec: dict, res: dict, stats: dict, patterns: list,
                mapping: dict, unite: str = "") -> dict:
    t = res["table"]
    points = [{"dimension": str(r["dim0"]), "value": round(float(r["value"]), 2)}
              for _, r in t.iterrows()]
    if "dim1" in t.columns:
        points = [{"dimension": str(r["dim0"]), "serie": str(r["dim1"]),
                   "value": round(float(r["value"]), 2)} for _, r in t.iterrows()]

    return {
        "spec_id": spec.get("id"),
        "source": spec.get("source"),
        "title": spec.get("title"),
        "business_question": spec.get("business_question"),
        "measure_label": spec["measure"].get("alias") or spec["measure"]["column"],
        "aggregation": spec["measure"]["agg"],
        "dimension_labels": res["dim_labels"],
        "unit": unite,
        "chart": spec.get("chart"),
        "data_points": points,
        "computed_stats": stats,
        "detected_patterns": patterns,
        "coverage": {
            "n_lignes_utilisees": res["n_lignes_utilisees"],
            "n_lignes_exclues": res["n_lignes_exclues"],
            "n_groupes_total": res["n_groupes_total"],
            "n_groupes_affiches": res["n_groupes_affiches"],
            "filtres": res["filtres_appliques"],
        },
    }


def run_plan(specs: list, df: pd.DataFrame, mapping: dict,
             unite: str = "") -> tuple[list, list]:
    """
    Exécute toutes les specs validées.

    Retourne (liste de facts, liste d'échecs). Une spec dont le résultat est
    vide ou trop pauvre est écartée ici (V9) — mieux vaut ne rien montrer
    qu'un graphique à un seul point.
    """
    inverse = {v: k for k, v in mapping.items() if v}

    contexte_global = {}
    if mapping.get("customer_id") in df.columns:
        contexte_global["n_clients_uniques"] = int(df[mapping["customer_id"]].nunique())
    if mapping.get("order_id") in df.columns:
        contexte_global["n_commandes"] = int(df[mapping["order_id"]].nunique())

    facts, echecs = [], []
    for spec in specs:
        try:
            res = execute_spec(spec, df)
            if res is None:
                echecs.append({"spec": spec.get("title"),
                               "reason": "V9 — résultat vide ou insuffisant"})
                continue
            stats = compute_stats(res)
            ctx = dict(contexte_global)
            ctx["business_role"] = inverse.get(spec["measure"]["column"])
            if spec["dimensions"]:
                ctx["dim_role"] = inverse.get(spec["dimensions"][0]["column"])
                if ctx.get("business_role") is None:
                    ctx["business_role"] = ctx["dim_role"]
            # Le rôle de la dimension sert aussi aux patterns géographiques
            if inverse.get(spec["dimensions"][0]["column"]) == "region":
                ctx["business_role"] = "region"
            patterns = detect_patterns(res, stats, spec, ctx)
            facts.append(build_facts(spec, res, stats, patterns, mapping, unite))
        except Exception as e:
            echecs.append({"spec": spec.get("title"),
                           "reason": f"{type(e).__name__} : {e}"})
    return facts, echecs


# --------------------------------------------------------------------------
# Périmètre de l'analyse
# --------------------------------------------------------------------------

def build_perimetre(df_raw, df_clean, mapping: dict,
                    cleaning_log: dict | None = None,
                    facts: list | None = None) -> dict:
    """
    Décrit SUR QUOI porte l'analyse : lignes retenues, période, montant.

    Un rapport qui affiche « 62 913 649 » sans dire s'il s'agit du chiffre
    d'affaires complet ou d'un sous-ensemble après correction est
    invérifiable — y compris par celui qui l'a produit. L'information existait
    déjà dans le champ `coverage` de chaque analyse, mais n'était affichée
    nulle part.
    """
    import pandas as pd

    from .profiler import clean_numeric_strings

    def num(df, col):
        if not col or col not in df.columns:
            return None
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            return s
        return pd.to_numeric(clean_numeric_strings(s), errors="coerce")

    rev = mapping.get("revenue")
    brut = num(df_raw, rev)
    net = num(df_clean, rev)

    date_col = mapping.get("order_date")
    periode = None
    if date_col and date_col in df_clean.columns:
        d = pd.to_datetime(df_clean[date_col], errors="coerce").dropna()
        if len(d):
            periode = {"debut": str(d.min().date()), "fin": str(d.max().date()),
                       "jours": int((d.max() - d.min()).days)}

    # Périodes de bord écartées des calculs de tendance
    exclues = []
    for f in (facts or []):
        for e in (f.get("computed_stats", {}).get("periodes_incompletes") or []):
            if e["periode"] not in [x["periode"] for x in exclues]:
                exclues.append(e)

    retirees = []
    for o in ((cleaning_log or {}).get("operations") or []):
        if o.get("rows_affected") and o["action"] in (
                "remove_rows", "remove_duplicates", "separate_rows"):
            retirees.append({"motif": o["issue_id"], "lignes": o["rows_affected"]})

    p = {
        "lignes_recues": int(len(df_raw)),
        "lignes_analysees": int(len(df_clean)),
        "lignes_ecartees": int(len(df_raw) - len(df_clean)),
        "part_analysee_pct": round(len(df_clean) / len(df_raw) * 100, 1) if len(df_raw) else 0.0,
        "periode": periode,
        "periodes_hors_tendance": exclues,
        "detail_ecarts": sorted(retirees, key=lambda x: -x["lignes"])[:6],
    }

    if brut is not None and net is not None:
        p["montant_recu"] = round(float(brut.sum()), 2)
        p["montant_analyse"] = round(float(net.sum()), 2)
        p["montant_ecarte"] = round(float(brut.sum() - net.sum()), 2)
        if brut.sum():
            p["part_montant_pct"] = round(float(net.sum() / brut.sum() * 100), 1)
        vides = int(net.isna().sum())
        if vides:
            p["lignes_sans_montant"] = vides

    return p


def phrase_perimetre(p: dict, unite: str = "") -> str:
    """Une phrase en clair, à placer en tête du rapport."""
    parts = [f"Analyse portant sur {p['lignes_analysees']:,} commandes"
             .replace(",", " ")]
    if p["lignes_ecartees"]:
        parts.append(f"sur les {p['lignes_recues']:,} de votre fichier "
                     f"({p['lignes_ecartees']:,} écartées)".replace(",", " "))
    if p.get("periode"):
        parts.append(f"du {p['periode']['debut']} au {p['periode']['fin']}")
    texte = ", ".join(parts) + "."

    if p.get("montant_ecarte") and p.get("part_montant_pct") is not None:
        pct = p["part_montant_pct"]
        base = f" Chiffre d'affaires retenu : {p['montant_analyse']:,.0f} {unite}"
        if pct > 100.5:
            # Retirer des montants NÉGATIFS (retours, avoirs) fait AUGMENTER
            # la somme. Annoncer « 109 % du total reçu » serait incompréhensible :
            # mieux vaut nommer la cause.
            base += (f", après retrait de {abs(p['montant_ecarte']):,.0f} {unite} "
                     f"de retours et avoirs.")
        else:
            base += f" ({pct} % du total reçu)."
        texte += base.replace(",", " ")
    elif p.get("montant_analyse") is not None:
        texte += f" Chiffre d'affaires : {p['montant_analyse']:,.0f} {unite}.".replace(",", " ")

    if p.get("lignes_sans_montant"):
        texte += (f" {p['lignes_sans_montant']:,} lignes n'ont pas de montant "
                  f"et ne comptent pas dans les totaux.").replace(",", " ")
    return texte
