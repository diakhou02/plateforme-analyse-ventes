"""
Vérification de la fidélité factuelle des interprétations.

Étape [8] du pipeline. C'est LE MODULE QUI PORTE LA PROMESSE DU TITRE.

Principe : extraire tous les nombres du texte généré, et vérifier que chacun
existe dans `facts.json`. Un nombre sans correspondance est une hallucination —
même si la phrase qui l'entoure est plausible.

Le résultat n'est pas seulement un contrôle : c'est une TRAÇABILITÉ. Chaque
nombre affiché à l'utilisateur peut être relié au champ exact qui l'a produit.

MÉTRIQUE PRINCIPALE DU MÉMOIRE
    fidélité = nombres vérifiés / nombres cités
À comparer avec le baseline « LLM brut recevant le CSV directement ».
"""

from __future__ import annotations

import re
import unicodedata

# Un nombre dans un texte français : « 1 234,56 », « 4820.56 », « −66,8 ».
#
# Les gardes (?<![\d.,]) et (?![\d]) sont indispensables : sans elles,
# l'alternance faisait correspondre « 4820.56 » au premier motif — conçu pour
# les séparateurs de milliers et limité à trois chiffres — qui s'arrêtait à
# « 482 » et laissait « 0.56 » comme second nombre. Deux hallucinations
# fabriquées de toutes pièces sur un chiffre parfaitement exact.
NOMBRE = re.compile(
    # Format américain : « 1,234.56 » — la virgule suivie d'exactement trois
    # chiffres ne peut pas être décimale, donc aucun risque avec « 66,8 ».
    r"(?<![\d.,])[-−+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d])"
    # Séparateur de milliers par espace : « 1 234,56 »
    r"|(?<![\d.,])[-−+]?\d{1,3}(?:[  \u00a0]\d{3})+(?:[.,]\d+)?(?![\d])"
    # Nombre simple : « 4820.56 », « 66,8 », « 1058 »
    r"|(?<![\d.,])[-−+]?\d+(?:[.,]\d+)?(?![\d])"
)

# Termes qui ne sont pas des mesures : dates, durées, quantités de langage
CONTEXTE_NON_NUMERIQUE = re.compile(
    r"\b(?:jan|f[ée]v|mar|avr|mai|juin|juil|ao[ûu]|sep|oct|nov|d[ée]c)\w*\b|"
    r"\b(?:lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\b|"
    r"\b(?:jours?|semaines?|mois|trimestres?|ann[ée]es?|heures?)\b",
    re.IGNORECASE,
)

TOLERANCE = 0.005          # ±0,5 % : arrondis d'affichage légitimes
PETITS_NOMBRES = {0, 1, 2, 3, 4, 5, 10, 100}   # « 3 actions », « 100 % »


def _normaliser(txt: str) -> float | None:
    """« 1 234,56 » → 1234.56"""
    t = (txt.replace("−", "-").replace("+", "")
            .replace("\u00a0", "").replace(" ", "").replace(" ", ""))
    if t.count(",") == 1 and "." not in t:
        t = t.replace(",", ".")
    else:
        t = t.replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


# Dates au format ISO présentes dans les libellés de période : « 2024-12 »,
# « 2024-12-31 ». Sans masquage préalable, l'expression régulière des nombres
# les découpe et produit des faux positifs (« 202 », « 12 ») comptés comme
# hallucinations.
DATE_ISO = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")


def extraire_nombres(texte: str) -> list[dict]:
    """Nombres cités dans un texte, avec leur contexte immédiat."""
    # Les dates sont remplacées par des caractères neutres de même longueur :
    # les positions restent valides pour l'extraction du contexte.
    masque = DATE_ISO.sub(lambda m: "·" * len(m.group()), texte)

    out = []
    for m in NOMBRE.finditer(masque):
        val = _normaliser(m.group())
        if val is None:
            continue
        debut, fin = max(0, m.start() - 40), min(len(texte), m.end() + 25)
        out.append({
            "texte": m.group(),
            "valeur": val,
            "contexte": texte[debut:fin].replace("\n", " "),
            "position": m.start(),
        })
    return out


def collecter_valeurs(facts: dict) -> list[dict]:
    """
    Toutes les valeurs vérifiables d'un bloc de facts, avec leur chemin.

    Le chemin est ce qui rend la traçabilité possible : on ne dit pas seulement
    « ce nombre existe », mais « il vient de computed_stats.change_pct ».
    """
    valeurs = []

    def ajouter(v, chemin):
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float)):
            valeurs.append({"valeur": float(v), "chemin": chemin})

    def parcourir(obj, prefixe=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                parcourir(v, f"{prefixe}.{k}" if prefixe else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                parcourir(v, f"{prefixe}[{i}]")
        else:
            ajouter(obj, prefixe)

    parcourir(facts.get("computed_stats", {}), "computed_stats")
    parcourir(facts.get("coverage", {}), "coverage")
    for i, p in enumerate(facts.get("detected_patterns", [])):
        parcourir(p.get("evidence", {}), f"pattern[{i}].evidence")
    for i, p in enumerate(facts.get("data_points", [])):
        ajouter(p.get("value"), f"data_points[{i}].value")

    # Valeurs dérivées ADMISES : le modèle peut légitimement écrire « −66,8 % »
    # là où le JSON contient -66.8, ou « une baisse de 66,8 % » sans le signe.
    derivees = []
    for v in valeurs:
        derivees.append({"valeur": abs(v["valeur"]), "chemin": v["chemin"] + " (valeur absolue)"})
        derivees.append({"valeur": round(v["valeur"]), "chemin": v["chemin"] + " (arrondi)"})
    return valeurs + derivees


def verifier_texte(texte: str, facts: dict) -> dict:
    """Vérifie un texte contre un bloc de facts."""
    cites = extraire_nombres(texte)
    connues = collecter_valeurs(facts)

    verifies, non_verifies, ignores = [], [], []

    for n in cites:
        val = n["valeur"]

        # Les petits nombres et les dates ne sont pas des mesures : les compter
        # comme hallucinations gonflerait artificiellement le taux d'erreur.
        if val in PETITS_NOMBRES and abs(val) <= 100:
            ignores.append({**n, "motif": "petit nombre courant"})
            continue
        if 1900 <= val <= 2100 and float(val).is_integer():
            ignores.append({**n, "motif": "année"})
            continue
        if CONTEXTE_NON_NUMERIQUE.search(n["contexte"]) and float(val).is_integer() and val <= 60:
            ignores.append({**n, "motif": "durée ou date"})
            continue

        correspondance = None
        for c in connues:
            ref = c["valeur"]
            if ref == 0:
                if abs(val) < 1e-9:
                    correspondance = c
                    break
                continue
            if abs(val - ref) / max(abs(ref), 1e-9) <= TOLERANCE:
                correspondance = c
                break

        if correspondance:
            verifies.append({**n, "source": correspondance["chemin"],
                             "valeur_source": correspondance["valeur"]})
        else:
            non_verifies.append(n)

    total = len(verifies) + len(non_verifies)
    return {
        "n_cites": len(cites),
        "n_verifies": len(verifies),
        "n_non_verifies": len(non_verifies),
        "n_ignores": len(ignores),
        "fidelite": round(len(verifies) / total, 3) if total else 1.0,
        "verifies": verifies,
        "non_verifies": non_verifies,
        "ignores": ignores,
    }


# --------------------------------------------------------------------------
# Contrôle du vocabulaire
# --------------------------------------------------------------------------

JARGON_INTERDIT = [
    "rfm", "cohorte", "pareto", "corrélation", "correlation", "écart-type",
    "ecart-type", "médiane", "mediane", "régression", "regression", "r²", "r2",
    "z-score", "quantile", "percentile", "variance", "outlier", "anomalie",
    "dataframe", "agrégation", "agregation", "cardinalité", "imputation",
    "linéaire", "lineaire", "significatif", "p-value",
]


def verifier_vocabulaire(texte: str) -> list[str]:
    """Repère le jargon technique qu'un commerçant ne comprendrait pas."""
    t = unicodedata.normalize("NFKD", texte.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    trouves = []
    for mot in JARGON_INTERDIT:
        m = unicodedata.normalize("NFKD", mot.lower())
        m = "".join(c for c in m if not unicodedata.combining(c))
        if re.search(rf"\b{re.escape(m)}\b", t):
            trouves.append(mot)
    return trouves


# --------------------------------------------------------------------------
# Vérification d'une interprétation complète
# --------------------------------------------------------------------------

def _racines(texte: str) -> set[str]:
    """
    Mots significatifs réduits à leur racine (5 premiers caractères).

    Une comparaison mot à mot échouait sur la conjugaison : le référentiel dit
    « Comparer vos prix », le modèle écrit « Comparez vos prix » — deux formes
    de la même action, comptées comme différentes. La troncature à 5 caractères
    rapproche « compar-er » et « compar-ez », « vérifi-er » et « vérifi-ez ».
    """
    t = unicodedata.normalize("NFKD", texte.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return {w[:5] for w in re.findall(r"\w{5,}", t)}


def verifier_interpretation(interpretation: dict, facts: dict) -> dict:
    """Contrôle les trois blocs, plus le vocabulaire et les actions."""
    if interpretation.get("error"):
        return {"error": interpretation["error"], "fidelite": None}

    texte_complet = " ".join([
        interpretation.get("constat", ""),
        interpretation.get("diagnostic", ""),
        " ".join(interpretation.get("actions", [])),
    ])

    fid = verifier_texte(texte_complet, facts)
    jargon = verifier_vocabulaire(texte_complet)

    # Les actions doivent dériver du pattern détecté, pas être inventées.
    refs = [_racines(a) for p in facts.get("detected_patterns", [])
            for a in p.get("actions", [])]
    actions_hors_ref = []
    for a in interpretation.get("actions", []):
        mots = _racines(a)
        if refs and not any(len(mots & r) >= 2 for r in refs):
            actions_hors_ref.append(a)

    statut = "ok"
    if fid["n_non_verifies"] > 0:
        statut = "chiffres non vérifiés"
    elif jargon:
        statut = "jargon détecté"
    elif actions_hors_ref:
        statut = "actions hors référentiel"

    return {
        "spec_id": facts.get("spec_id"),
        "statut": statut,
        "fidelite": fid["fidelite"],
        "n_chiffres_cites": fid["n_cites"],
        "n_verifies": fid["n_verifies"],
        "n_non_verifies": fid["n_non_verifies"],
        "chiffres_suspects": fid["non_verifies"],
        "tracabilite": [{"texte": v["texte"], "source": v["source"]}
                        for v in fid["verifies"]],
        "jargon": jargon,
        "actions_hors_referentiel": actions_hors_ref,
    }


def rapport_global(verifications: list) -> dict:
    """Agrège les vérifications — c'est le résultat principal du mémoire."""
    valides = [v for v in verifications if v.get("fidelite") is not None]
    if not valides:
        return {"error": "aucune interprétation vérifiable"}

    cites = sum(v["n_chiffres_cites"] for v in valides)
    ok = sum(v["n_verifies"] for v in valides)
    ko = sum(v["n_non_verifies"] for v in valides)

    return {
        "n_interpretations": len(valides),
        "n_chiffres_cites": cites,
        "n_verifies": ok,
        "n_non_verifies": ko,
        "fidelite_globale": round(ok / (ok + ko), 4) if (ok + ko) else 1.0,
        "n_avec_jargon": sum(1 for v in valides if v["jargon"]),
        "n_actions_hors_referentiel": sum(len(v["actions_hors_referentiel"])
                                          for v in valides),
        "interpretations_parfaites": sum(1 for v in valides if v["statut"] == "ok"),
    }
