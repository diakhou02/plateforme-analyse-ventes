"""
Validation des spécifications produites par le LLM.

Étape [6a] du pipeline. Filtre `plan.json` avant exécution.

POURQUOI CE MODULE N'EST PAS OPTIONNEL
Un LLM inventera des noms de colonnes, proposera des agrégations absurdes et
oubliera des contraintes explicitement énoncées dans le prompt. C'est
systématique, pas accidentel — aucune formulation de prompt ne l'élimine.

Le validateur est la frontière entre « le modèle propose » et « le système
exécute ». Sans lui, une colonne hallucinée fait planter l'exécution, ou pire :
une agrégation illégitime produit un chiffre faux mais plausible.

Le taux de specs valides est aussi une MÉTRIQUE D'ÉVALUATION du mémoire.
"""

from __future__ import annotations

from .planner import AGGREGATIONS, CHARTS, DIM_TYPES, FILTER_OPS, GRAINS

REJET, CORRECTION = "rejected", "corrected"


class SpecValidator:
    """Applique les règles V1 à V10. Rejette ou corrige, ne devine jamais."""

    def __init__(self, profile: dict, df=None):
        self.cols = {c["name"]: c for c in profile["columns"]}
        self.df = df

    # ------------------------------------------------------------------

    def validate(self, spec: dict) -> tuple[dict | None, list[dict]]:
        """Retourne (spec corrigée, journal). Spec None si rejetée."""
        journal: list[dict] = []
        s = dict(spec)

        def rejeter(regle: str, motif: str):
            journal.append({"rule": regle, "action": REJET, "reason": motif,
                            "spec_id": s.get("id")})
            return None, journal

        def corriger(regle: str, motif: str):
            journal.append({"rule": regle, "action": CORRECTION, "reason": motif,
                            "spec_id": s.get("id")})

        # --- forme générale ---
        mesure = s.get("measure") or {}
        dims = s.get("dimensions") or []
        if not mesure.get("column"):
            return rejeter("V0", "aucune mesure")
        if not 1 <= len(dims) <= 2:
            return rejeter("V0", f"{len(dims)} dimension(s), 1 ou 2 attendues")

        # --- V1 : les colonnes existent ---
        # La règle la plus souvent déclenchée. Le modèle invente des noms
        # plausibles ou reprend un nom vu dans un autre jeu de données.
        col_m = mesure["column"]
        if col_m not in self.cols:
            return rejeter("V1", f"colonne de mesure inconnue : « {col_m} »")
        for d in dims:
            if d.get("column") not in self.cols:
                return rejeter("V1", f"colonne de dimension inconnue : « {d.get('column')} »")

        info_m = self.cols[col_m]
        agg = mesure.get("agg")

        # --- énumérations ---
        if agg not in AGGREGATIONS:
            return rejeter("V0", f"agrégation inconnue : {agg}")
        for d in dims:
            if d.get("type") not in DIM_TYPES:
                return rejeter("V0", f"type de dimension inconnu : {d.get('type')}")
            if d.get("grain") and d["grain"] not in GRAINS:
                corriger("V0", f"granularité inconnue ({d['grain']}), passée à month")
                d["grain"] = "month"

        # --- V2 : ne pas sommer une mesure non additive ---
        # Sommer un taux ou une note produit un nombre dénué de sens, mais
        # parfaitement affichable — donc invisible sans ce contrôle.
        if agg == "sum" and info_m.get("additive") is False:
            corriger("V2", f"« {col_m} » n'est pas sommable, agrégation passée à mean")
            mesure["agg"] = agg = "mean"

        # --- V3 : pas d'agrégation numérique sur un identifiant ---
        if info_m["role_candidate"] == "identifier" and agg not in {"count", "count_distinct"}:
            corriger("V3", f"« {col_m} » est un identifiant, agrégation passée à count_distinct")
            mesure["agg"] = agg = "count_distinct"

        # Une mesure doit être numérique, sauf en comptage
        if (info_m["role_candidate"] not in {"measure", "identifier"}
                and agg not in {"count", "count_distinct"}):
            return rejeter("V3", f"« {col_m} » n'est pas une mesure ({info_m['role_candidate']})")

        # --- V4 : une dimension temporelle doit l'être vraiment ---
        for d in dims:
            info_d = self.cols[d["column"]]
            if d["type"] == "temporal" and info_d["role_candidate"] != "temporal":
                return rejeter("V4", f"« {d['column']} » n'est pas une colonne de date")
            if d["type"] == "temporal" and not d.get("grain"):
                corriger("V4", "granularité manquante, month par défaut")
                d["grain"] = "month"

        # --- V5 : plafonner les dimensions à forte cardinalité ---
        # Sans limite, un graphique à 3000 barres — illisible et coûteux.
        principale = self.cols[dims[0]["column"]]
        if (dims[0]["type"] != "temporal" and principale["n_unique"] > 50
                and not s.get("limit")):
            corriger("V5", f"« {dims[0]['column']} » a {principale['n_unique']} valeurs, "
                           f"limite fixée à 15")
            s["limit"] = 15

        # --- V6 : line exige une dimension temporelle ---
        chart = s.get("chart")
        if chart not in CHARTS:
            corriger("V6", f"graphique inconnu ({chart}), passé à bar")
            s["chart"] = chart = "bar"
        temporelle = any(d["type"] == "temporal" for d in dims)
        if chart in {"line", "area"} and not temporelle:
            corriger("V6", "graphique en courbe sans axe temporel, passé à bar")
            s["chart"] = "bar"

        # --- V7 : scatter exige deux dimensions numériques ---
        if chart == "scatter" and len(dims) < 2:
            corriger("V7", "nuage de points avec une seule dimension, passé à bar")
            s["chart"] = "bar"

        # --- V10 : filtres sur des valeurs inexistantes ---
        filtres_ok = []
        for f in s.get("filters") or []:
            col_f = f.get("column")
            if col_f not in self.cols:
                corriger("V10", f"filtre sur colonne inconnue « {col_f} », supprimé")
                continue
            if f.get("op") not in FILTER_OPS:
                corriger("V10", f"opérateur inconnu ({f.get('op')}), filtre supprimé")
                continue
            if self.df is not None and f["op"] in {"eq", "in"}:
                presentes = set(self.df[col_f].dropna().astype(str).unique())
                attendues = f["value"] if isinstance(f["value"], list) else [f["value"]]
                gardees = [v for v in attendues if str(v) in presentes]
                if not gardees:
                    corriger("V10", f"filtre sur des valeurs absentes de « {col_f} », supprimé")
                    continue
                if len(gardees) < len(attendues):
                    corriger("V10", f"valeurs absentes retirées du filtre sur « {col_f} »")
                    f["value"] = gardees if isinstance(f["value"], list) else gardees[0]
            filtres_ok.append(f)
        s["filters"] = filtres_ok

        # --- normalisation ---
        s["measure"] = mesure
        s["dimensions"] = dims
        mesure.setdefault("alias", f"{agg}_{col_m}")
        s.setdefault("sort", {"by": "dimension" if temporelle else "measure",
                              "order": "asc" if temporelle else "desc"})
        return s, journal


def _signature(spec: dict) -> tuple:
    """
    Identité SÉMANTIQUE d'une spec, pour la déduplication (V8).

    `limit` et `sort` en sont volontairement exclus : « CA par catégorie,
    top 15 » et « CA par catégorie » sont la MÊME analyse. Les inclure faisait
    passer des doublons du socle pour des propositions nouvelles, gaspillant
    les places disponibles pendant que les vraies découvertes du modèle
    étaient écartées.

    Les filtres, eux, comptent : « CA par catégorie » et « CA par catégorie,
    hors annulées » répondent à deux questions différentes.
    """
    m = spec["measure"]
    filtres = tuple(sorted(
        (f.get("column"), f.get("op"), str(f.get("value")))
        for f in (spec.get("filters") or [])
    ))
    return (m["column"], m["agg"],
            tuple(sorted((d["column"], d.get("grain")) for d in spec["dimensions"])),
            filtres)


def validate_plan(plan: dict, profile: dict, df=None,
                  max_specs: int = 8, min_llm_specs: int = 3) -> dict:
    """
    Valide, déduplique et priorise le plan.

    Deux garanties symétriques :
    - le socle passe en premier — ces analyses doivent apparaître même si le
      modèle a produit des propositions mieux notées ;
    - `min_llm_specs` places sont réservées aux découvertes du modèle.

    Sans cette réserve, le socle (6 analyses) monopolisait 6 des 8 places, et
    les croisements originaux du modèle — ceux qui justifient son emploi —
    étaient systématiquement écartés. Un socle qui étouffe le planificateur
    vide l'architecture de son intérêt.
    """
    validateur = SpecValidator(profile, df)
    valides, rejets, journal = [], [], []
    vues = set()

    for spec in plan.get("specs", []):
        s, log = validateur.validate(spec)
        journal += log
        if s is None:
            rejets.append({"spec": spec.get("title") or spec.get("id"),
                           "source": spec.get("source"),
                           "reason": log[-1]["reason"] if log else "invalide"})
            continue

        # V8 — déduplication : le modèle reformule souvent une analyse du socle.
        sig = _signature(s)
        if sig in vues:
            rejets.append({"spec": s.get("title"), "source": s.get("source"),
                           "reason": "doublon d'une analyse déjà retenue"})
            journal.append({"rule": "V8", "action": REJET,
                            "reason": "spec identique", "spec_id": s.get("id")})
            continue
        vues.add(sig)
        valides.append(s)

    socle = [s for s in valides if s.get("source") == "socle"]
    llm = sorted([s for s in valides if s.get("source") != "socle"],
                 key=lambda x: x.get("priority", 99))

    n_llm = min(len(llm), max(min_llm_specs, max_specs - len(socle)))
    n_socle = max(0, max_specs - n_llm)
    retenues = socle[:n_socle] + llm[:n_llm]

    total = len(plan.get("specs", []))
    return {
        "specs": retenues,
        "all_valid_specs": valides,
        "rejected": rejets,
        "validation_log": journal,
        "stats": {
            "n_proposed": total,
            "n_valid": len(valides),
            "n_rejected": len(rejets),
            "n_corrected": len({e["spec_id"] for e in journal
                                if e["action"] == CORRECTION}),
            "n_executed": len(retenues),
            "n_from_socle": len(socle[:n_socle]),
            "n_from_llm": len(llm[:n_llm]),
            "validity_rate": round(len(valides) / total, 3) if total else 0.0,
        },
    }
