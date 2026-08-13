"""
Détection des anomalies de qualité dans un fichier de vente.

Étape [3] du pipeline. Ne corrige RIEN : produit `quality_report.json`,
le diagnostic que l'utilisateur validera avant nettoyage (étape [4]).

Principe : détecter tout, classer chaque anomalie selon sa politique
(AUTO / AUTO_NOTIFIED / ASK / BLOCK), et n'agir sans demander que sur
ce qui ne peut, dans aucun scénario métier, faire perdre une information.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .profiler import clean_numeric_strings, normalize_name

# --------------------------------------------------------------------------
# Politiques (§1 de la spécification)
# --------------------------------------------------------------------------
AUTO = "AUTO"                    # corrigé sans demander
AUTO_NOTIFIED = "AUTO_NOTIFIED"  # corrigé, mis en avant, annulable
ASK = "ASK"                      # rien n'est fait avant réponse
BLOCK = "BLOCK"                  # analyse impossible

SEV_HIGH, SEV_MED, SEV_LOW = "high", "medium", "low"

TOTAL_TOKENS = {"total", "totaux", "somme", "sous-total", "sous total",
                "subtotal", "grand total", "cumul", "moyenne"}

TEST_TOKENS = {"test", "essai", "demo", "dummy", "aaa", "xxx", "zzz",
               "azerty", "qwerty", "toto", "sample", "exemple"}

CANCELLED = {"annule", "annulee", "cancelled", "canceled", "refunded",
             "rembourse", "remboursee", "unavailable", "failed", "echec",
             "returned", "retourne", "void"}

NULL_TOKENS = {"", "na", "n/a", "nan", "none", "null", "-", "--", "?", "#n/a",
               "#na", "inconnu", "non renseigne", "vide", "nil", "undefined"}


def _strip_accents(s: str) -> str:
    n = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in n if not unicodedata.combining(c))


def _levenshtein(a: str, b: str, cap: int = 3) -> int:
    """Distance d'édition, abandonnée au-delà de `cap` (suffisant pour C04)."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------------
# Structure d'une anomalie
# --------------------------------------------------------------------------

@dataclass
class Issue:
    id: str
    severity: str
    policy: str
    user_message: str
    user_explanation: str = ""
    impact_if_ignored: str = ""
    columns: list = field(default_factory=list)
    affected_rows: int = 0
    affected_share: float = 0.0
    recommendation: str | None = None
    recommendation_label: str = ""
    options: list = field(default_factory=list)
    details: dict = field(default_factory=dict)
    sample_rows: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "severity": self.severity,
            "policy": self.policy,
            "columns": self.columns,
            "affected_rows": int(self.affected_rows),
            "affected_share": round(float(self.affected_share), 4),
            "user_message": self.user_message,
            "user_explanation": self.user_explanation,
        }
        if self.impact_if_ignored:
            d["impact_if_ignored"] = self.impact_if_ignored
        if self.recommendation:
            d["recommendation"] = self.recommendation
            d["recommendation_label"] = self.recommendation_label
        if self.options:
            d["options"] = self.options
        if self.details:
            d["details"] = self.details
        if self.sample_rows:
            d["sample_rows"] = self.sample_rows[:10]
        return d


def _opts(*pairs) -> list:
    """Construit la liste d'options ; la première est le choix recommandé."""
    return [{"action": a, "label": l, "is_default": i == 0}
            for i, (a, l) in enumerate(pairs)]


# ==========================================================================
# S — Structure du fichier
# ==========================================================================

def detect_structure(df: pd.DataFrame, profile: dict, read_report) -> list[Issue]:
    out: list[Issue] = []
    n = len(df)

    # S01/S02/S04/S09 sont détectés à la lecture : on les reprend tels quels.
    if read_report:
        for raw in getattr(read_report, "issues", []):
            out.append(Issue(
                id=raw["id"],
                severity=SEV_LOW,
                policy=raw.get("policy", AUTO),
                user_message=raw["message"],
                user_explanation="Votre fichier a été ouvert correctement malgré ce point.",
            ))

    # S05 / S06 — colonnes vides ou constantes
    for c in profile["columns"]:
        if c["role_candidate"] == "empty":
            out.append(Issue(
                id="S05_empty_columns", severity=SEV_LOW, policy=AUTO,
                columns=[c["name"]], affected_rows=n, affected_share=1.0,
                user_message=f"La colonne « {c['name']} » est entièrement vide.",
                user_explanation="Elle ne contient aucune information et a été retirée.",
            ))
        elif c["role_candidate"] == "constant":
            out.append(Issue(
                id="S06_constant_columns", severity=SEV_LOW, policy=AUTO_NOTIFIED,
                columns=[c["name"]], affected_rows=n, affected_share=1.0,
                user_message=f"La colonne « {c['name']} » contient toujours la même valeur.",
                user_explanation="Une colonne identique partout ne permet aucune comparaison. "
                                 "Elle a été mise de côté.",
                details={"valeur": (c["top_values"][0]["value"] if c["top_values"] else None)},
            ))

    # S07 — lignes de total glissées dans le tableau
    mask = pd.Series(False, index=df.index)
    for col in df.columns:
        s = df[col].astype(str).str.strip().str.lower().map(_strip_accents)
        mask |= s.isin({_strip_accents(t) for t in TOTAL_TOKENS})
    if mask.any():
        rows = df.index[mask].tolist()
        out.append(Issue(
            id="S07_total_rows", severity=SEV_HIGH, policy=AUTO_NOTIFIED,
            affected_rows=len(rows), affected_share=len(rows) / n,
            sample_rows=[int(i) for i in rows],
            user_message=f"{len(rows)} ligne(s) de total se trouvent dans votre tableau.",
            user_explanation="Ces lignes récapitulatives ne sont pas des ventes réelles. "
                             "Si on les garde, elles sont comptées comme des commandes.",
            impact_if_ignored="Votre chiffre d'affaires serait compté deux fois.",
            recommendation="remove_total_rows",
            recommendation_label="Retirer ces lignes",
            options=_opts(("remove_total_rows", "Retirer"), ("keep_all", "Garder"),
                          ("preview", "Voir les lignes")),
        ))
    return out


# ==========================================================================
# D — Doublons
# ==========================================================================

def detect_duplicates(df: pd.DataFrame, profile: dict, mapping: dict | None) -> list[Issue]:
    out: list[Issue] = []
    n = len(df)

    # D01 — lignes strictement identiques
    dup = df.duplicated(keep="first")
    if dup.any():
        k = int(dup.sum())
        out.append(Issue(
            id="D01_exact_duplicates", severity=SEV_HIGH, policy=ASK,
            affected_rows=k, affected_share=k / n,
            sample_rows=[int(i) for i in df.index[dup][:10]],
            user_message=f"{k} commande(s) apparaissent plusieurs fois à l'identique.",
            user_explanation="Cela arrive souvent quand un fichier est exporté ou importé "
                             "deux fois. Ces copies gonflent artificiellement vos résultats.",
            impact_if_ignored=f"Votre chiffre d'affaires serait surestimé d'environ "
                              f"{k / n * 100:.1f} %.",
            recommendation="remove_duplicates",
            recommendation_label="Ne garder qu'un exemplaire de chaque",
            options=_opts(("remove_duplicates", "Corriger"), ("keep_all", "Tout garder"),
                          ("preview", "Voir les lignes")),
        ))

    # D02 — même identifiant de commande sur plusieurs lignes
    #
    # PIÈGE MAJEUR : dans un fichier « une ligne = un article », plusieurs
    # lignes partagent légitimement le même order_id. Dédupliquer là-dessus
    # détruirait le chiffre d'affaires. On ne signale que si TOUTES les
    # autres colonnes sont également identiques.
    order_col = (mapping or {}).get("order_id")
    if order_col and order_col in df.columns:
        dup_id = df[order_col].duplicated(keep=False) & df[order_col].notna()
        if dup_id.any():
            sub = df[dup_id]
            fully = sub.duplicated(keep="first").sum()
            n_ids = int(sub[order_col].nunique())
            if fully > 0:
                out.append(Issue(
                    id="D02_id_duplicates", severity=SEV_MED, policy=ASK,
                    columns=[order_col], affected_rows=int(fully),
                    affected_share=float(fully) / n,
                    user_message=f"{fully} ligne(s) répètent un numéro de commande "
                                 f"avec exactement les mêmes informations.",
                    user_explanation="Il s'agit probablement d'un double enregistrement.",
                    recommendation="remove_duplicates",
                    recommendation_label="Ne garder qu'un exemplaire",
                    options=_opts(("remove_duplicates", "Corriger"), ("keep_all", "Garder")),
                    details={"commandes_concernees": n_ids},
                ))
            else:
                # Cas normal et fréquent : on informe, on ne propose RIEN.
                out.append(Issue(
                    id="D02_multiline_orders", severity=SEV_LOW, policy=AUTO,
                    columns=[order_col], affected_rows=int(dup_id.sum()),
                    affected_share=float(dup_id.sum()) / n,
                    user_message="Vos commandes contiennent plusieurs articles chacune.",
                    user_explanation="C'est normal : une ligne correspond à un article. "
                                     "Les totaux seront calculés par commande.",
                    details={"commandes_multi_articles": n_ids},
                ))

    # D03 — quasi-doublons : même client, même date, même montant
    m = mapping or {}
    keys = [m.get("customer_id"), m.get("order_date"), m.get("revenue")]
    keys = [k for k in keys if k and k in df.columns]
    if len(keys) == 3:
        near = df.duplicated(subset=keys, keep="first") & ~df.duplicated(keep="first")
        if near.any():
            k = int(near.sum())
            out.append(Issue(
                id="D03_near_duplicates", severity=SEV_MED, policy=ASK,
                columns=keys, affected_rows=k, affected_share=k / n,
                sample_rows=[int(i) for i in df.index[near][:10]],
                user_message=f"{k} commande(s) semblent enregistrées deux fois.",
                user_explanation="Même client, même jour, même montant, mais un numéro "
                                 "différent. C'est souvent le signe d'un double import.",
                recommendation="review",
                recommendation_label="Vérifier avant de décider",
                options=_opts(("preview", "Voir les lignes"),
                              ("remove_duplicates", "Fusionner"), ("keep_all", "Garder")),
            ))
    return out


# ==========================================================================
# T — Dates
# ==========================================================================

def detect_dates(df: pd.DataFrame, profile: dict) -> list[Issue]:
    out: list[Issue] = []
    n = len(df)

    for c in profile["columns"]:
        if c["role_candidate"] != "temporal":
            continue
        col, st = c["name"], c["stats"]
        s = pd.to_datetime(df[col], errors="coerce", format="mixed", dayfirst=True)

        # T02 — jour et mois indiscernables
        vals = df[col].dropna().astype(str)
        parts = vals.str.extract(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.]")
        if parts.notna().all(axis=1).sum() > len(vals) * 0.8:
            a = pd.to_numeric(parts[0], errors="coerce")
            b = pd.to_numeric(parts[1], errors="coerce")
            if a.max() <= 12 and b.max() <= 12:
                out.append(Issue(
                    id="T02_ambiguous_format", severity=SEV_HIGH, policy=ASK,
                    columns=[col], affected_rows=n, affected_share=1.0,
                    user_message=f"Les dates de « {col} » peuvent se lire de deux façons.",
                    user_explanation="Par exemple, 10/03/2024 peut être le 10 mars ou le "
                                     "3 octobre. Le choix change vos résultats mensuels.",
                    recommendation="dayfirst",
                    recommendation_label="Jour/Mois/Année (format français)",
                    options=_opts(("dayfirst", "Jour/Mois/Année"),
                                  ("monthfirst", "Mois/Jour/Année")),
                ))

        # T03 — dates futures
        future = int((s > pd.Timestamp.now()).sum())
        if future:
            out.append(Issue(
                id="T03_future_dates", severity=SEV_MED, policy=ASK,
                columns=[col], affected_rows=future, affected_share=future / n,
                user_message=f"{future} commande(s) portent une date future.",
                user_explanation="Il peut s'agir de précommandes, ou d'une erreur de saisie.",
                recommendation="keep",
                recommendation_label="Garder (précommandes possibles)",
                options=_opts(("keep", "Garder"), ("exclude", "Exclure"),
                              ("preview", "Voir les lignes")),
            ))

        # T04 — dates absurdes (1900 = valeur sentinelle Excel)
        absurd = int(((s.dt.year < 1990) | (s.dt.year == 1900)).sum())
        if absurd:
            out.append(Issue(
                id="T04_absurd_dates", severity=SEV_MED, policy=AUTO_NOTIFIED,
                columns=[col], affected_rows=absurd, affected_share=absurd / n,
                user_message=f"{absurd} date(s) sont manifestement erronées.",
                user_explanation="Des dates antérieures à 1990 apparaissent quand une "
                                 "cellule est vide dans un tableur. Elles ont été ignorées.",
            ))

        # T01 — non parsables
        bad = int(s.isna().sum() - df[col].isna().sum())
        if bad > n * 0.02:
            out.append(Issue(
                id="T01_unparsable", severity=SEV_HIGH, policy=ASK,
                columns=[col], affected_rows=bad, affected_share=bad / n,
                user_message=f"{bad} date(s) n'ont pas pu être lues dans « {col} ».",
                user_explanation="Le format de ces dates est inhabituel.",
                recommendation="exclude",
                recommendation_label="Ignorer ces lignes",
                options=_opts(("exclude", "Ignorer"), ("preview", "Voir les lignes")),
            ))

        # T06 — période trop courte
        if st.get("span_days", 999) < 56:
            out.append(Issue(
                id="T06_insufficient_span", severity=SEV_MED, policy=AUTO_NOTIFIED,
                columns=[col], affected_rows=n, affected_share=1.0,
                user_message="Votre fichier couvre une période trop courte.",
                user_explanation=f"Il faut au moins deux mois de données pour repérer une "
                                 f"tendance. Vous en avez {st.get('span_days')} jours.",
                impact_if_ignored="Les analyses d'évolution et de saisonnalité sont désactivées.",
            ))
    return out


# ==========================================================================
# M — Montants et quantités
# ==========================================================================

def detect_measures(df: pd.DataFrame, profile: dict) -> list[Issue]:
    out: list[Issue] = []
    n = len(df)

    for c in profile["columns"]:
        if c["role_candidate"] != "measure":
            continue
        col, st = c["name"], c["stats"]
        s = pd.to_numeric(clean_numeric_strings(df[col]), errors="coerce")

        # M07 — séparateur décimal ambigu (détecté au profilage)
        if "M07_ambiguous_decimal" in c["flags"]:
            out.append(Issue(
                id="M07_ambiguous_decimal", severity=SEV_HIGH, policy=ASK,
                columns=[col], affected_rows=n, affected_share=1.0,
                user_message=f"Les nombres de « {col} » peuvent se lire de deux façons.",
                user_explanation="Par exemple, 1,250 peut valoir 1,25 ou 1250. "
                                 "Le choix change complètement vos totaux.",
                recommendation="decimal",
                recommendation_label="La virgule sépare les décimales (1,25)",
                options=_opts(("decimal", "1,250 = 1,25"), ("thousands", "1,250 = 1250")),
            ))

        # M04 / M06 — nettoyage de forme, sans risque
        if df[col].dtype == object:
            has_sym = df[col].astype(str).str.contains(r"[€$£¥]|R\$", regex=True).mean()
            if has_sym > 0.5:
                out.append(Issue(
                    id="M04_currency_symbols", severity=SEV_LOW, policy=AUTO,
                    columns=[col], affected_rows=n, affected_share=1.0,
                    user_message=f"Les montants de « {col} » contenaient un symbole monétaire.",
                    user_explanation="Ils ont été convertis en nombres pour permettre les calculs.",
                ))

        # M01 — valeurs négatives
        neg = int(st.get("n_negative", 0))
        if neg:
            out.append(Issue(
                id="M01_negative", severity=SEV_MED, policy=ASK,
                columns=[col], affected_rows=neg, affected_share=neg / n,
                user_message=f"{neg} ligne(s) ont un montant négatif dans « {col} ».",
                user_explanation="Ce sont peut-être des remboursements. Si vous les gardez, "
                                 "ils viendront en déduction de votre chiffre d'affaires.",
                recommendation="separate",
                recommendation_label="Les analyser à part",
                options=_opts(("separate", "Analyser à part"), ("keep", "Garder dans le total"),
                              ("exclude", "Exclure"), ("preview", "Voir les lignes")),
            ))

        # M02 — valeurs nulles
        zero = int(st.get("n_zero", 0))
        if zero and zero / n >= 0.01:
            out.append(Issue(
                id="M02_zero_values", severity=SEV_LOW, policy=ASK,
                columns=[col], affected_rows=zero, affected_share=zero / n,
                user_message=f"{zero} ligne(s) ont un montant à zéro dans « {col} ».",
                user_explanation="Cadeaux, échantillons, ou commandes de test. "
                                 "Gardées, elles font baisser votre panier moyen.",
                recommendation="exclude",
                recommendation_label="Les exclure des moyennes",
                options=_opts(("exclude", "Exclure"), ("keep", "Garder"),
                              ("preview", "Voir les lignes")),
            ))

        # M03 — valeurs extrêmes
        v = s.dropna()
        if len(v) > 20:
            med = v.median()
            mad = (v - med).abs().median()
            if mad > 0:
                # Score robuste : la médiane n'est pas tirée par les extrêmes,
                # contrairement à la moyenne utilisée par le z-score classique.
                rz = 0.6745 * (v - med).abs() / mad
                ext = v[(rz > 10) & (v.abs() > med * 20)]
                if len(ext):
                    out.append(Issue(
                        id="M03_extreme_outliers", severity=SEV_HIGH, policy=ASK,
                        columns=[col], affected_rows=len(ext), affected_share=len(ext) / n,
                        sample_rows=[int(i) for i in ext.index[:10]],
                        user_message=f"{len(ext)} montant(s) sont anormalement élevés "
                                     f"dans « {col} ».",
                        user_explanation=f"La valeur la plus haute est {ext.max():,.0f}, "
                                         f"alors que la commande habituelle est autour de "
                                         f"{med:,.0f}. Cela ressemble à une erreur de saisie "
                                         f"ou à une ligne de total.",
                        impact_if_ignored="Vos moyennes seront fortement faussées.",
                        recommendation="preview",
                        recommendation_label="Vérifier ces lignes",
                        options=_opts(("preview", "Voir les lignes"), ("exclude", "Exclure"),
                                      ("keep", "Garder")),
                        details={"valeur_max": float(ext.max()), "mediane": float(med)},
                    ))
    return out


# ==========================================================================
# C — Catégories et libellés
# ==========================================================================

def detect_categories(df: pd.DataFrame, profile: dict) -> list[Issue]:
    out: list[Issue] = []
    n = len(df)

    for c in profile["columns"]:
        if c["role_candidate"] not in {"categorical", "boolean_flag"}:
            continue
        col = c["name"]
        s = df[col].dropna().astype(str)
        if s.empty:
            continue

        # C01 — espaces parasites
        ws = int((s != s.str.strip()).sum())
        if ws:
            out.append(Issue(
                id="C01_whitespace", severity=SEV_LOW, policy=AUTO,
                columns=[col], affected_rows=ws, affected_share=ws / n,
                user_message=f"Des espaces inutiles entouraient certaines valeurs de « {col} ».",
                user_explanation="Sans correction, « Beauté » et « Beauté » (avec un espace) "
                                 "seraient comptés séparément.",
            ))

        t = s.str.strip()

        # C05 — marqueurs textuels de valeur manquante
        nulls = int(t.str.lower().isin(NULL_TOKENS).sum())
        if nulls:
            out.append(Issue(
                id="C05_null_placeholders", severity=SEV_LOW, policy=AUTO,
                columns=[col], affected_rows=nulls, affected_share=nulls / n,
                user_message=f"{nulls} valeur(s) de « {col} » signifiaient « non renseigné ».",
                user_explanation="Des mentions comme « N/A » ou « - » ont été traitées "
                                 "comme des cases vides.",
            ))

        # C02 — variantes de casse
        groups: dict[str, list[str]] = {}
        for v in t.unique():
            groups.setdefault(v.lower(), []).append(v)
        case_var = {k: g for k, g in groups.items() if len(g) > 1}
        if case_var:
            vc = t.value_counts()
            details = []
            touched = 0
            for g in case_var.values():
                kept = max(g, key=lambda x: vc.get(x, 0))
                rows = int(sum(vc.get(x, 0) for x in g if x != kept))
                touched += rows
                details.append({"conserve": kept,
                                "fusionnes": [x for x in g if x != kept],
                                "lignes": rows})
            out.append(Issue(
                id="C02_case_variants", severity=SEV_MED, policy=AUTO_NOTIFIED,
                columns=[col], affected_rows=touched, affected_share=touched / n,
                user_message=f"{len(case_var)} valeur(s) de « {col} » étaient écrites "
                             f"de plusieurs façons.",
                user_explanation="Par exemple « Beauté » et « BEAUTÉ » étaient comptées "
                                 "comme deux catégories différentes. Elles ont été regroupées.",
                details={"regroupements": details},
            ))

        # C03 — variantes d'accentuation
        acc: dict[str, list[str]] = {}
        for v in t.unique():
            acc.setdefault(_strip_accents(v).lower(), []).append(v)
        acc_var = {k: g for k, g in acc.items()
                   if len(g) > 1 and len({x.lower() for x in g}) > 1}
        if acc_var:
            out.append(Issue(
                id="C03_accent_variants", severity=SEV_MED, policy=ASK,
                columns=[col], affected_rows=len(acc_var), affected_share=0.0,
                user_message=f"Certaines valeurs de « {col} » ne diffèrent que "
                             f"par les accents.",
                user_explanation="Par exemple « beaute » et « beauté ». S'agit-il de "
                                 "la même chose ?",
                recommendation="merge",
                recommendation_label="Les regrouper",
                options=_opts(("merge", "Regrouper"), ("keep_separate", "Garder séparées")),
                details={"groupes": [{"valeurs": g} for g in list(acc_var.values())[:10]]},
            ))

        # C04 — fautes de frappe (distance d'édition faible)
        uniq = t.value_counts()
        if 2 <= len(uniq) <= 200:
            names = list(uniq.index)
            pairs, deja = [], set()
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    if a.lower() == b.lower() or b in deja:
                        continue
                    if abs(len(a) - len(b)) <= 2 and _levenshtein(a.lower(), b.lower()) <= 2:
                        # La forme la plus fréquente est retenue comme correcte.
                        # On ne conditionne PAS à un écart de fréquence : deux
                        # orthographes peuvent coexister à parts égales dans un
                        # fichier saisi à la main. C'est ASK, donc l'utilisateur
                        # tranche — au système de le signaler, pas de décider.
                        maj, mino = (a, b) if uniq[a] >= uniq[b] else (b, a)
                        pairs.append({"probable": maj, "variante": mino,
                                      "lignes": int(uniq[mino])})
                        deja.add(mino)
            if pairs:
                touched = sum(p["lignes"] for p in pairs)
                exemple = pairs[0]
                out.append(Issue(
                    id="C04_near_duplicates", severity=SEV_MED, policy=ASK,
                    columns=[col], affected_rows=touched, affected_share=touched / n,
                    user_message=f"{len(pairs)} valeur(s) de « {col} » ressemblent à "
                                 f"des fautes de frappe.",
                    user_explanation=f"Par exemple « {exemple['variante']} » au lieu de "
                                     f"« {exemple['probable']} ». Non corrigées, elles "
                                     f"apparaissent comme des catégories distinctes et "
                                     f"divisent vos ventes en deux.",
                    recommendation="merge",
                    recommendation_label="Corriger les fautes",
                    options=_opts(("merge", "Corriger"), ("keep_separate", "Garder telles quelles"),
                                  ("preview", "Voir le détail")),
                    details={"paires": pairs[:15]},
                ))

        # C06 — cardinalité excessive
        if c["n_unique"] > 200:
            out.append(Issue(
                id="C06_high_cardinality", severity=SEV_LOW, policy=AUTO_NOTIFIED,
                columns=[col], affected_rows=n, affected_share=1.0,
                user_message=f"« {col} » contient trop de valeurs différentes "
                             f"({c['n_unique']}).",
                user_explanation="Un graphique avec autant de barres serait illisible. "
                                 "Seules les principales seront affichées.",
            ))
    return out


# ==========================================================================
# N — Valeurs manquantes
# ==========================================================================

def detect_missing(df: pd.DataFrame, profile: dict, mapping: dict | None) -> list[Issue]:
    out: list[Issue] = []
    n = len(df)
    m = mapping or {}
    critical = {v for k, v in m.items()
                if k in {"revenue", "order_date", "order_id"} and v}

    for c in profile["columns"]:
        col, rate = c["name"], c["null_rate"]
        if rate <= 0 or c["role_candidate"] in {"empty", "constant"}:
            continue
        k = int(round(rate * n))

        if col in critical:
            out.append(Issue(
                id="N01_critical_missing", severity=SEV_HIGH, policy=ASK,
                columns=[col], affected_rows=k, affected_share=rate,
                user_message=f"{k} ligne(s) n'ont pas d'information dans « {col} ».",
                user_explanation="Cette colonne est indispensable au calcul. Les lignes "
                                 "concernées ne pourront pas être analysées.",
                impact_if_ignored=f"{rate * 100:.1f} % de vos ventes seraient absentes "
                                  f"des résultats.",
                recommendation="exclude",
                recommendation_label="Ignorer ces lignes",
                options=_opts(("exclude", "Ignorer"), ("keep", "Garder"),
                              ("preview", "Voir les lignes")),
            ))
        elif rate > 0.5:
            out.append(Issue(
                id="N02_high_null_rate", severity=SEV_MED, policy=AUTO_NOTIFIED,
                columns=[col], affected_rows=k, affected_share=rate,
                user_message=f"« {col} » est vide dans {rate * 100:.0f} % des cas.",
                user_explanation="Trop incomplète pour être fiable, cette colonne ne sera "
                                 "pas utilisée dans les analyses principales.",
            ))
        elif rate > 0.05:
            out.append(Issue(
                id="N03_moderate_missing", severity=SEV_LOW, policy=AUTO_NOTIFIED,
                columns=[col], affected_rows=k, affected_share=rate,
                user_message=f"« {col} » est vide dans {rate * 100:.0f} % des cas.",
                user_explanation="Les lignes concernées seront écartées uniquement des "
                                 "calculs utilisant cette colonne.",
            ))
    return out


# ==========================================================================
# X — Lignes de test et statuts
# ==========================================================================

def detect_special_rows(df: pd.DataFrame, profile: dict, mapping: dict | None) -> list[Issue]:
    out: list[Issue] = []
    n = len(df)

    # X01 — lignes de test
    mask = pd.Series(False, index=df.index)
    for c in profile["columns"]:
        if c["role_candidate"] in {"categorical", "identifier", "high_cardinality_cat", "text"}:
            s = df[c["name"]].astype(str).str.lower()
            for tok in TEST_TOKENS:
                mask |= s.str.fullmatch(rf"\s*{tok}\s*", na=False)
                mask |= s.str.contains(rf"\b{tok}\b", regex=True, na=False)
    if mask.any():
        k = int(mask.sum())
        out.append(Issue(
            id="X01_test_rows", severity=SEV_MED, policy=ASK,
            affected_rows=k, affected_share=k / n,
            sample_rows=[int(i) for i in df.index[mask][:10]],
            user_message=f"{k} ligne(s) ressemblent à des commandes de test.",
            user_explanation="Elles contiennent des mentions comme « test » ou « demo ». "
                             "Ce ne sont probablement pas de vraies ventes.",
            recommendation="exclude",
            recommendation_label="Les exclure",
            options=_opts(("exclude", "Exclure"), ("keep", "Garder"),
                          ("preview", "Voir les lignes")),
        ))

    # X02 — commandes annulées ou remboursées
    status_col = (mapping or {}).get("order_status")
    if not status_col:
        for c in profile["columns"]:
            if c["role_candidate"] == "categorical" and c["top_values"]:
                vals = {_strip_accents(str(v["value"])).lower() for v in c["top_values"]}
                if vals & CANCELLED:
                    status_col = c["name"]
                    break
    if status_col and status_col in df.columns:
        s = df[status_col].astype(str).str.strip().str.lower().map(_strip_accents)
        mask = s.isin({_strip_accents(x) for x in CANCELLED})
        if mask.any():
            k = int(mask.sum())
            out.append(Issue(
                id="X02_cancelled_orders", severity=SEV_HIGH, policy=ASK,
                columns=[status_col], affected_rows=k, affected_share=k / n,
                user_message=f"{k} commande(s) sont annulées ou remboursées "
                             f"({k / n * 100:.1f} %).",
                user_explanation="Ces commandes n'ont pas généré de revenu réel. Les inclure "
                                 "gonfle votre chiffre d'affaires.",
                impact_if_ignored=f"Votre chiffre d'affaires serait surestimé d'environ "
                                  f"{k / n * 100:.1f} %.",
                recommendation="separate",
                recommendation_label="Les retirer du chiffre d'affaires et les analyser à part",
                options=_opts(("separate", "Retirer du CA"), ("keep", "Tout garder"),
                              ("exclude", "Supprimer complètement")),
                details={"statuts": sorted(df.loc[mask, status_col].astype(str).unique().tolist())},
            ))

    # X03 — client hyperactif (compte interne ou revendeur)
    cust = (mapping or {}).get("customer_id")
    if cust and cust in df.columns and df[cust].notna().sum() > 50:
        vc = df[cust].value_counts()
        if len(vc) > 1 and vc.iloc[0] / n >= 0.05:
            out.append(Issue(
                id="X03_internal_orders", severity=SEV_LOW, policy=ASK,
                columns=[cust], affected_rows=int(vc.iloc[0]),
                affected_share=float(vc.iloc[0]) / n,
                user_message=f"Un même client représente {vc.iloc[0] / n * 100:.0f} % "
                             f"de vos commandes.",
                user_explanation="Il peut s'agir d'un compte interne, d'un revendeur, ou de "
                                 "commandes sans identification client. Cela fausserait "
                                 "l'analyse de fidélité.",
                recommendation="keep",
                recommendation_label="Garder mais signaler",
                options=_opts(("keep", "Garder"), ("exclude", "Exclure ce client")),
                details={"client": str(vc.index[0]), "commandes": int(vc.iloc[0])},
            ))
    return out


# ==========================================================================
# B — Blocages
# ==========================================================================

def detect_blockers(df: pd.DataFrame, profile: dict) -> list[Issue]:
    out: list[Issue] = []
    roles = profile["roles_summary"]

    if len(df) < 30:
        out.append(Issue(
            id="B01_no_rows", severity=SEV_HIGH, policy=BLOCK,
            affected_rows=len(df), affected_share=1.0,
            user_message="Votre fichier contient trop peu de commandes.",
            user_explanation=f"Il en faut au moins 30 pour une analyse fiable. "
                             f"Vous en avez {len(df)}.",
        ))
    if not roles.get("measure"):
        out.append(Issue(
            id="B02_no_measure", severity=SEV_HIGH, policy=BLOCK,
            user_message="Aucune colonne de montant n'a été trouvée.",
            user_explanation="Vérifiez que votre fichier contient bien les prix ou "
                             "les quantités vendues.",
        ))
    if not roles.get("temporal"):
        out.append(Issue(
            id="B03_no_date", severity=SEV_MED, policy=AUTO_NOTIFIED,
            user_message="Aucune date n'a été trouvée dans votre fichier.",
            user_explanation="L'analyse reste possible, mais sans évolution dans le temps "
                             "ni saisonnalité.",
        ))
    return out


# ==========================================================================
# Score de qualité
# ==========================================================================

def compute_quality_score(df: pd.DataFrame, profile: dict, issues: list[Issue]) -> dict:
    """
    Quatre dimensions classiques de la qualité des données, moyennées,
    puis pénalisées selon la GRAVITÉ des anomalies non résolues.

    La proportion de lignes touchées ne suffit pas : une seule ligne de total
    peut fausser toutes les moyennes, tandis que 10 % de valeurs manquantes sur
    une colonne secondaire ne gêne presque rien. Sans cette pénalité, un fichier
    à corriger d'urgence obtient « très bon état ».
    """
    n = len(df)
    cols = [c for c in profile["columns"] if c["role_candidate"] != "empty"]

    completeness = 1.0 - (float(np.mean([c["null_rate"] for c in cols])) if cols else 0.0)

    def share(ids: set[str]) -> float:
        rows = sum(i.affected_rows for i in issues if i.id in ids)
        return min(1.0, rows / n) if n else 0.0

    consistency = 1.0 - share({"C01_whitespace", "C02_case_variants", "C03_accent_variants",
                               "C04_near_duplicates", "S07_total_rows"})
    validity = 1.0 - share({"T01_unparsable", "T03_future_dates", "T04_absurd_dates",
                            "M01_negative", "M03_extreme_outliers", "X01_test_rows",
                            "M07_ambiguous_decimal", "X02_cancelled_orders"})
    uniqueness = 1.0 - share({"D01_exact_duplicates", "D02_id_duplicates",
                              "D03_near_duplicates"})

    dims = {
        "completeness": round(max(0.0, completeness), 3),
        "consistency": round(max(0.0, consistency), 3),
        "validity": round(max(0.0, validity), 3),
        "uniqueness": round(max(0.0, uniqueness), 3),
    }

    base = float(np.mean(list(dims.values()))) * 100

    # Pénalité de gravité : chaque anomalie exigeant un arbitrage coûte des
    # points indépendamment du nombre de lignes concernées.
    penalite = 0
    for i in issues:
        if i.policy == BLOCK:
            penalite += 40
        elif i.policy == ASK:
            penalite += {SEV_HIGH: 9, SEV_MED: 5, SEV_LOW: 2}[i.severity]
        elif i.policy == AUTO_NOTIFIED and i.severity == SEV_HIGH:
            penalite += 4

    score = int(round(max(0.0, min(100.0, base - penalite))))
    return {"score": score, "breakdown": dims, "penalty": penalite}


def score_label(score: int) -> str:
    if score >= 90:
        return "Votre fichier est en très bon état."
    if score >= 75:
        return "Votre fichier est en bon état, quelques points à vérifier."
    if score >= 55:
        return "Votre fichier demande quelques corrections avant analyse."
    if score >= 30:
        return "Plusieurs problèmes fausseraient vos résultats. Corrigez-les avant d'analyser."
    return "Votre fichier contient de nombreux problèmes à corriger."


# ==========================================================================
# Point d'entrée
# ==========================================================================

def diagnose(df: pd.DataFrame, profile: dict, read_report=None,
             mapping: dict | None = None) -> dict:
    """Produit `quality_report.json`. Ne modifie jamais `df`."""
    issues: list[Issue] = []
    issues += detect_blockers(df, profile)
    issues += detect_structure(df, profile, read_report)
    issues += detect_duplicates(df, profile, mapping)
    issues += detect_dates(df, profile)
    issues += detect_measures(df, profile)
    issues += detect_categories(df, profile)
    issues += detect_missing(df, profile, mapping)
    issues += detect_special_rows(df, profile, mapping)
    issues += detect_coherence(df, profile, mapping)

    q = compute_quality_score(df, profile, issues)
    order = {SEV_HIGH: 0, SEV_MED: 1, SEV_LOW: 2}
    issues.sort(key=lambda i: (order[i.severity], -i.affected_share))

    by_policy: dict[str, list] = {}
    for i in issues:
        by_policy.setdefault(i.policy, []).append(i.to_dict())

    return {
        "file": {
            "name": getattr(read_report, "filename", None),
            "rows_raw": int(len(df)),
            "columns_raw": int(df.shape[1]),
            "encoding_detected": getattr(read_report, "encoding", None),
            "separator_detected": getattr(read_report, "separator", None),
        },
        "quality_score": q["score"],
        "quality_label": score_label(q["score"]),
        "score_breakdown": q["breakdown"],
        "n_issues": len(issues),
        "n_decisions_required": len(by_policy.get(ASK, [])),
        "blocked": by_policy.get(BLOCK, []),
        "decisions_required": by_policy.get(ASK, []),
        "auto_notified": by_policy.get(AUTO_NOTIFIED, []),
        "auto_applied": by_policy.get(AUTO, []),
        "issues": [i.to_dict() for i in issues],
    }


# ==========================================================================
# R — Cohérence entre colonnes
# ==========================================================================
#
# C'est ce qui distingue un nettoyage automatique d'un nettoyage d'analyste.
# Les familles précédentes examinent chaque colonne ISOLÉMENT : une valeur
# manquante, un doublon, un format. Un analyste, lui, croise les colonnes :
# « quantité × prix unitaire donne-t-il bien le montant ? », « la livraison
# est-elle postérieure à la commande ? »
#
# Ces incohérences sont invisibles colonne par colonne, et pourtant elles
# révèlent les erreurs les plus graves — celles qui faussent le chiffre
# d'affaires sans qu'aucune valeur ne paraisse anormale.

def detect_coherence(df: pd.DataFrame, profile: dict,
                     mapping: dict | None) -> list[Issue]:
    out: list[Issue] = []
    m = mapping or {}
    n = len(df)
    if n == 0:
        return out

    def num(cle):
        col = m.get(cle)
        if col and col in df.columns:
            return col, pd.to_numeric(clean_numeric_strings(df[col]), errors="coerce")
        return None, None

    def dt(cle):
        col = m.get(cle)
        if col and col in df.columns:
            return col, pd.to_datetime(df[col], errors="coerce",
                                       format="mixed", dayfirst=True)
        return None, None

    # --- R01 : quantité × prix unitaire ≈ montant -------------------------
    c_qte, qte = num("quantity")
    c_pu, pu = num("unit_price")
    c_rev, rev = num("revenue")

    if qte is not None and pu is not None and rev is not None:
        attendu = qte * pu
        valides = attendu.notna() & rev.notna() & (attendu != 0)
        ecart = (rev - attendu).abs() / attendu.abs().where(attendu != 0)
        mask = valides & (ecart > 0.02)          # 2 % de tolérance : arrondis
        k = int(mask.sum())
        if k and k / n > 0.005:
            exemples = df.index[mask][:5]
            out.append(Issue(
                id="R01_amount_mismatch", severity=SEV_HIGH, policy=ASK,
                columns=[c_qte, c_pu, c_rev], affected_rows=k, affected_share=k / n,
                sample_rows=[int(i) for i in exemples],
                user_message=f"{k} ligne(s) ont un montant qui ne correspond pas "
                             f"à la quantité multipliée par le prix.",
                user_explanation="Par exemple, une commande de 3 articles à 10 € "
                                 "devrait afficher 30 €. Cet écart vient souvent "
                                 "d'une remise non enregistrée, de frais ajoutés, "
                                 "ou d'une erreur de saisie.",
                impact_if_ignored="Impossible de savoir quel chiffre est juste : "
                                  "votre chiffre d'affaires pourrait être faux.",
                recommendation="preview",
                recommendation_label="Voir les lignes avant de décider",
                options=_opts(("preview", "Voir les lignes"),
                              ("recompute", "Recalculer le montant"),
                              ("keep", "Garder tel quel")),
                details={"ecart_median_pct": round(float(ecart[mask].median() * 100), 1)},
            ))

    # --- R02 : livraison postérieure à la commande -----------------------
    c_cmd, d_cmd = dt("order_date")
    c_liv, d_liv = dt("delivery_date")

    if d_cmd is not None and d_liv is not None:
        mask = d_cmd.notna() & d_liv.notna() & (d_liv < d_cmd)
        k = int(mask.sum())
        if k:
            out.append(Issue(
                id="R02_delivery_before_order", severity=SEV_HIGH, policy=ASK,
                columns=[c_cmd, c_liv], affected_rows=k, affected_share=k / n,
                sample_rows=[int(i) for i in df.index[mask][:5]],
                user_message=f"{k} commande(s) sont livrées avant d'avoir été passées.",
                user_explanation="C'est impossible. Les deux dates ont probablement "
                                 "été inversées, ou l'une des deux est mal saisie.",
                recommendation="preview",
                recommendation_label="Voir les lignes concernées",
                options=_opts(("preview", "Voir les lignes"),
                              ("swap", "Inverser les deux dates"),
                              ("exclude", "Ignorer ces lignes")),
            ))

        # Délai de livraison aberrant : plus d'un an
        delai = (d_liv - d_cmd).dt.days
        mask_long = delai > 365
        k = int(mask_long.sum())
        if k and k / n > 0.005:
            out.append(Issue(
                id="R03_delivery_delay", severity=SEV_MED, policy=AUTO_NOTIFIED,
                columns=[c_cmd, c_liv], affected_rows=k, affected_share=k / n,
                user_message=f"{k} commande(s) affichent un délai de livraison "
                             f"de plus d'un an.",
                user_explanation="Ces délais sont probablement dus à une erreur de "
                                 "date. Ils sont signalés mais conservés.",
                details={"delai_max_jours": int(delai.max())},
            ))

    # --- R04 : la somme des lignes égale-t-elle le total ? ---------------
    c_ord = m.get("order_id")
    c_tot = m.get("order_total")
    if (c_ord and c_ord in df.columns and c_tot and c_tot in df.columns
            and rev is not None):
        tot = pd.to_numeric(clean_numeric_strings(df[c_tot]), errors="coerce")
        somme = rev.groupby(df[c_ord]).transform("sum")
        premier = ~df[c_ord].duplicated()
        ecart = (tot - somme).abs() / somme.abs().where(somme != 0)
        mask = premier & somme.notna() & tot.notna() & (ecart > 0.02)
        k = int(mask.sum())
        if k:
            out.append(Issue(
                id="R04_total_mismatch", severity=SEV_HIGH, policy=ASK,
                columns=[c_ord, c_tot], affected_rows=k, affected_share=k / n,
                sample_rows=[int(i) for i in df.index[mask][:5]],
                user_message=f"{k} commande(s) ont un total différent de la somme "
                             f"de leurs articles.",
                user_explanation="Le total de la commande devrait correspondre à "
                                 "l'addition de ses articles. L'écart vient souvent "
                                 "des frais de port ou d'une remise globale.",
                impact_if_ignored="Selon la colonne utilisée, votre chiffre d'affaires "
                                  "changera.",
                recommendation="preview",
                recommendation_label="Voir les commandes concernées",
                options=_opts(("preview", "Voir les commandes"),
                              ("use_lines", "Utiliser la somme des articles"),
                              ("use_total", "Utiliser le total indiqué")),
            ))

    # --- R05 : prix unitaire implicite aberrant --------------------------
    # Un prix unitaire déduit peut être absurde alors que le montant et la
    # quantité paraissent normaux chacun de leur côté.
    if qte is not None and rev is not None and pu is None:
        valides = qte.notna() & rev.notna() & (qte > 0)
        implicite = (rev / qte).where(valides)
        v = implicite.dropna()
        if len(v) > 30:
            med = float(v.median())
            mad = float((v - med).abs().median())
            if mad > 0 and med > 0:
                score = 0.6745 * (implicite - med).abs() / mad
                mask = valides & (score > 15) & ((implicite < med / 50) |
                                                 (implicite > med * 50))
                k = int(mask.sum())
                if k and k / n > 0.002:
                    out.append(Issue(
                        id="R05_implied_price_outlier", severity=SEV_MED, policy=ASK,
                        columns=[c_qte, c_rev], affected_rows=k, affected_share=k / n,
                        sample_rows=[int(i) for i in df.index[mask][:5]],
                        user_message=f"{k} ligne(s) donnent un prix par article "
                                     f"très inhabituel.",
                        user_explanation=f"Le prix par article y est très éloigné de "
                                         f"vos {med:,.0f} habituels. Cela arrive quand "
                                         f"la quantité ou le montant a été mal saisi."
                                         .replace(",", " "),
                        recommendation="preview",
                        recommendation_label="Vérifier ces lignes",
                        options=_opts(("preview", "Voir les lignes"),
                                      ("exclude", "Exclure"), ("keep", "Garder")),
                        details={"prix_habituel": round(med, 2)},
                    ))

    # --- R06 : statut incohérent avec les montants -----------------------
    c_st = m.get("order_status")
    if c_st and c_st in df.columns and rev is not None:
        s = df[c_st].astype(str).str.strip().str.lower().map(_strip_accents)
        annulees = s.isin({_strip_accents(x) for x in CANCELLED})
        mask = annulees & rev.notna() & (rev > 0)
        k = int(mask.sum())
        if k and k / n > 0.01:
            montant = float(rev[mask].sum())
            out.append(Issue(
                id="R06_cancelled_with_amount", severity=SEV_HIGH, policy=ASK,
                columns=[c_st, c_rev], affected_rows=k, affected_share=k / n,
                user_message=f"{k} commande(s) annulées portent quand même un montant.",
                user_explanation="Une commande annulée n'a pas généré de revenu. "
                                 "Si ces montants restent comptés, votre chiffre "
                                 "d'affaires est surévalué.",
                impact_if_ignored=f"Environ {montant:,.0f} de trop dans votre "
                                  f"chiffre d'affaires.".replace(",", " "),
                recommendation="zero_out",
                recommendation_label="Mettre ces montants à zéro",
                options=_opts(("zero_out", "Mettre à zéro"),
                              ("exclude", "Retirer ces lignes"),
                              ("keep", "Garder")),
                details={"montant_concerne": round(montant, 2)},
            ))

    # --- R07 : frais de port supérieurs au montant -----------------------
    c_frais, frais = num("shipping_cost")
    if frais is not None and rev is not None:
        mask = frais.notna() & rev.notna() & (rev > 0) & (frais > rev * 2)
        k = int(mask.sum())
        if k and k / n > 0.005:
            out.append(Issue(
                id="R07_shipping_exceeds_amount", severity=SEV_MED, policy=AUTO_NOTIFIED,
                columns=[c_frais, c_rev], affected_rows=k, affected_share=k / n,
                user_message=f"{k} commande(s) ont des frais de livraison bien "
                             f"supérieurs au prix des articles.",
                user_explanation="C'est possible pour des produits lourds ou lointains, "
                                 "mais cela peut aussi signaler une erreur. Ces lignes "
                                 "sont signalées, pas modifiées.",
            ))

    return out
