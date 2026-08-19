"""
Nettoyage des données à partir des décisions de l'utilisateur.

Étape [4] du pipeline. Consomme `quality_report.json` et produit un
DataFrame nettoyé accompagné de `cleaning_log.json`.

PRINCIPE DE RÉVERSIBILITÉ
Le fichier d'origine n'est jamais écrasé. Les corrections forment une pile
d'opérations rejouables : `df_clean` est TOUJOURS recalculé depuis `df_raw`
en appliquant la pile dans l'ordre. Annuler une opération revient à la retirer
de la pile et à tout rejouer — c'est plus simple et bien plus sûr que de tenter
d'inverser une opération.

RÈGLE ABSOLUE : aucune imputation. On ne remplace jamais une valeur manquante
par une moyenne ou une médiane. Inventer une donnée contredirait frontalement
la promesse de vérifiabilité. Les lignes incomplètes sont exclues du calcul
concerné, et le rapport indique combien.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .profiler import clean_numeric_strings
from .quality import (ASK, AUTO, AUTO_NOTIFIED, BLOCK, CANCELLED, NULL_TOKENS,
                      TOTAL_TOKENS, _strip_accents)


# --------------------------------------------------------------------------
# Une opération de la pile
# --------------------------------------------------------------------------

@dataclass
class Operation:
    seq: int
    issue_id: str
    action: str
    trigger: str                       # "auto" | "user_confirmed" | "user_default"
    params: dict = field(default_factory=dict)
    columns: list = field(default_factory=list)
    rows_affected: int = 0
    rows_removed: list = field(default_factory=list)
    timestamp: str = ""
    undoable: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        d = {
            "seq": self.seq,
            "issue_id": self.issue_id,
            "action": self.action,
            "trigger": self.trigger,
            "params": self.params,
            "columns": self.columns,
            "rows_affected": int(self.rows_affected),
            "timestamp": self.timestamp,
            "undoable": self.undoable,
        }
        if self.rows_removed:
            d["rows_removed_index"] = [int(i) for i in self.rows_removed[:500]]
        if self.note:
            d["note"] = self.note
        return d


# --------------------------------------------------------------------------
# Le nettoyeur
# --------------------------------------------------------------------------

class Cleaner:
    """
    Applique les corrections de façon réversible.

    Utilisation :
        c = Cleaner(df_raw, quality_report, mapping)
        c.apply_defaults()                    # recommandations du système
        c.set_decision("D01_exact_duplicates", "keep_all")   # arbitrage utilisateur
        df_clean = c.result()
    """

    def __init__(self, df_raw: pd.DataFrame, report: dict, mapping: dict | None = None):
        self.df_raw = df_raw
        self.report = report
        self.mapping = mapping or {}

        # Regroupement par identifiant, PAS un simple dict : une même anomalie
        # peut toucher plusieurs colonnes (C01 sur trois catégories, par exemple).
        # Un dict {id: issue} écraserait silencieusement toutes les occurrences
        # sauf la dernière — et les colonnes concernées ne seraient pas nettoyées.
        self.issues_by_id: dict[str, list[dict]] = {}
        for i in report.get("issues", []):
            self.issues_by_id.setdefault(i["id"], []).append(i)

        self.decisions: dict[str, str] = {}
        self.operations: list[Operation] = []
        self._cache: pd.DataFrame | None = None

    @property
    def issues(self) -> dict[str, dict]:
        """Vue « une entrée par identifiant », colonnes fusionnées."""
        out = {}
        for iid, lst in self.issues_by_id.items():
            base = dict(lst[0])
            cols, samples = [], []
            for i in lst:
                cols += [c for c in i.get("columns", []) if c not in cols]
                samples += i.get("sample_rows", [])
            base["columns"] = cols
            base["sample_rows"] = samples
            out[iid] = base
        return out

    # ---------------------------------------------------------------- API

    def apply_defaults(self) -> "Cleaner":
        """Applique la recommandation du système pour chaque anomalie ASK."""
        for iid, issue in self.issues.items():
            if issue["policy"] == ASK and issue.get("recommendation"):
                self.decisions[iid] = issue["recommendation"]
        self._cache = None
        return self

    def set_decision(self, issue_id: str, action: str) -> "Cleaner":
        """
        Enregistre l'arbitrage de l'utilisateur pour une anomalie.

        Une décision portant sur une anomalie absente est IGNORÉE, jamais
        fatale : elle provient presque toujours d'un fichier analysé
        précédemment dans la même session. Faire tomber toute l'application
        pour une décision devenue sans objet serait disproportionné.
        """
        if issue_id not in self.issues:
            return self
        self.decisions[issue_id] = action
        self._cache = None
        return self

    def undo(self, issue_id: str) -> "Cleaner":
        """Annule une correction : on la retire et on rejoue toute la pile."""
        self.decisions.pop(issue_id, None)
        self.decisions[issue_id] = "keep_all"
        self._cache = None
        return self

    def result(self) -> pd.DataFrame:
        """DataFrame nettoyé. Toujours recalculé depuis df_raw."""
        if self._cache is None:
            self._cache = self._replay()
        return self._cache

    # ------------------------------------------------------------ Rejeu

    def _replay(self) -> pd.DataFrame:
        df = self.df_raw.copy()
        self.operations = []
        self._seq = 0

        # L'ordre compte : on retire d'abord ce qui n'est pas une vraie vente,
        # on normalise ensuite, on convertit en dernier.
        for etape in (self._structure, self._rows, self._categories,
                      self._measures, self._dates, self._missing):
            df = etape(df)
        return df.reset_index(drop=True)

    def _log(self, issue_id: str, action: str, rows: int = 0, cols=None,
             removed=None, params=None, note: str = "") -> None:
        self._seq += 1
        issue = self.issues.get(issue_id, {})
        trigger = ("auto" if issue.get("policy") in (AUTO, AUTO_NOTIFIED)
                   else ("user_confirmed" if issue_id in self.decisions else "auto"))
        self.operations.append(Operation(
            seq=self._seq, issue_id=issue_id, action=action, trigger=trigger,
            params=params or {}, columns=cols or [], rows_affected=rows,
            rows_removed=list(removed or []),
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            note=note,
        ))

    def _decided(self, issue_id: str) -> str | None:
        """Action retenue pour une anomalie, ou None si aucune."""
        if issue_id not in self.issues:
            return None
        issue = self.issues[issue_id]
        if issue["policy"] in (AUTO, AUTO_NOTIFIED):
            return "auto"
        return self.decisions.get(issue_id)

    # ------------------------------------------------------- S — structure

    def _structure(self, df: pd.DataFrame) -> pd.DataFrame:
        # S05 / S06 — colonnes vides ou constantes
        for iid in ("S05_empty_columns", "S06_constant_columns"):
            issue = self.issues.get(iid)
            if issue and self._decided(iid):
                cols = [c for c in issue["columns"] if c in df.columns]
                if cols:
                    df = df.drop(columns=cols)
                    self._log(iid, "drop_columns", cols=cols,
                              note="Colonne sans information analysable")

        # S07 — lignes de total
        if self._decided("S07_total_rows") in ("auto", "remove_total_rows"):
            mask = pd.Series(False, index=df.index)
            for col in df.columns:
                s = df[col].astype(str).str.strip().str.lower().map(_strip_accents)
                mask |= s.isin({_strip_accents(t) for t in TOTAL_TOKENS})
            if mask.any():
                removed = df.index[mask].tolist()
                df = df[~mask]
                self._log("S07_total_rows", "remove_rows", rows=len(removed),
                          removed=removed, note="Lignes récapitulatives, pas des ventes")
        return df

    # ------------------------------------------------------- X / D — lignes

    def _rows(self, df: pd.DataFrame) -> pd.DataFrame:
        # X01 — lignes de test
        act = self._decided("X01_test_rows")
        if act == "exclude":
            idx = self.issues["X01_test_rows"].get("sample_rows", [])
            mask = df.index.isin(idx)
            if mask.any():
                df = df[~mask]
                self._log("X01_test_rows", "remove_rows", rows=int(mask.sum()),
                          removed=list(np.array(idx)))

        # X02 — commandes annulées ou remboursées
        act = self._decided("X02_cancelled_orders")
        if act in ("separate", "exclude"):
            issue = self.issues["X02_cancelled_orders"]
            col = issue["columns"][0] if issue["columns"] else None
            if col and col in df.columns:
                s = df[col].astype(str).str.strip().str.lower().map(_strip_accents)
                mask = s.isin({_strip_accents(x) for x in CANCELLED})
                if mask.any():
                    removed = df.index[mask].tolist()
                    # "separate" conserve les lignes dans un attribut dédié :
                    # elles restent analysables à part, hors chiffre d'affaires.
                    if act == "separate":
                        self.cancelled = df[mask].copy()
                    df = df[~mask]
                    self._log("X02_cancelled_orders",
                              "separate_rows" if act == "separate" else "remove_rows",
                              rows=len(removed), cols=[col], removed=removed,
                              note="Retirées du chiffre d'affaires")

        # D01 — doublons stricts
        if self._decided("D01_exact_duplicates") == "remove_duplicates":
            dup = df.duplicated(keep="first")
            if dup.any():
                removed = df.index[dup].tolist()
                df = df[~dup]
                self._log("D01_exact_duplicates", "remove_duplicates",
                          rows=len(removed), removed=removed,
                          params={"subset": "*", "keep": "first"})

        # D02 — même numéro de commande, lignes strictement identiques
        #
        # Ne s'exécute JAMAIS sur D02_multiline_orders : dans un fichier
        # « une ligne = un article », dédupliquer sur le numéro de commande
        # détruirait le chiffre d'affaires.
        if self._decided("D02_id_duplicates") == "remove_duplicates":
            dup = df.duplicated(keep="first")
            if dup.any():
                removed = df.index[dup].tolist()
                df = df[~dup]
                self._log("D02_id_duplicates", "remove_duplicates",
                          rows=len(removed), removed=removed)

        # D03 — quasi-doublons
        if self._decided("D03_near_duplicates") == "remove_duplicates":
            keys = [self.mapping.get(k) for k in ("customer_id", "order_date", "revenue")]
            keys = [k for k in keys if k and k in df.columns]
            if len(keys) == 3:
                dup = df.duplicated(subset=keys, keep="first")
                if dup.any():
                    removed = df.index[dup].tolist()
                    df = df[~dup]
                    self._log("D03_near_duplicates", "remove_duplicates",
                              rows=len(removed), cols=keys, removed=removed)

        # X03 — client hyperactif : on signale, on ne retire que sur demande
        if self._decided("X03_internal_orders") == "exclude":
            issue = self.issues["X03_internal_orders"]
            col = issue["columns"][0] if issue["columns"] else None
            client = (issue.get("details") or {}).get("client")
            if col and col in df.columns and client is not None:
                mask = df[col].astype(str) == str(client)
                if mask.any():
                    removed = df.index[mask].tolist()
                    df = df[~mask]
                    self._log("X03_internal_orders", "remove_rows",
                              rows=len(removed), cols=[col], removed=removed,
                              params={"client": client})
        return df

    # ------------------------------------------------------- C — catégories

    def _categories(self, df: pd.DataFrame) -> pd.DataFrame:
        # C01 — espaces parasites
        if self._decided("C01_whitespace"):
            cols = [c for c in self.issues["C01_whitespace"]["columns"] if c in df.columns]
            for col in cols:
                df[col] = df[col].astype(str).str.strip().where(df[col].notna())
            if cols:
                self._log("C01_whitespace", "strip_whitespace", cols=cols)

        # C05 — marqueurs textuels de valeur manquante
        if self._decided("C05_null_placeholders"):
            cols = [c for c in self.issues["C05_null_placeholders"]["columns"]
                    if c in df.columns]
            total = 0
            for col in cols:
                mask = df[col].astype(str).str.strip().str.lower().isin(NULL_TOKENS)
                total += int(mask.sum())
                df.loc[mask, col] = np.nan
            if cols:
                self._log("C05_null_placeholders", "to_null", rows=total, cols=cols,
                          note="« N/A », « - » traités comme cases vides")

        # C02 — variantes de casse : on retient la forme la plus fréquente
        issue = self.issues.get("C02_case_variants")
        if issue and self._decided("C02_case_variants"):
            for col in [c for c in issue["columns"] if c in df.columns]:
                s = df[col].astype(str).str.strip()
                vc = s.value_counts()
                repl = {}
                groups: dict[str, list[str]] = {}
                for v in s.dropna().unique():
                    groups.setdefault(v.lower(), []).append(v)
                for g in groups.values():
                    if len(g) > 1:
                        kept = max(g, key=lambda x: vc.get(x, 0))
                        repl.update({x: kept for x in g if x != kept})
                if repl:
                    mask = s.isin(repl)
                    df[col] = s.replace(repl).where(df[col].notna())
                    self._log("C02_case_variants", "normalize_case",
                              rows=int(mask.sum()), cols=[col],
                              params={"remplacements": len(repl)})

        # C03 / C04 — accents et fautes de frappe : uniquement sur demande.
        # On parcourt chaque occurrence, les détails (paires, groupes) étant
        # propres à chaque colonne.
        for iid, action_ok in (("C03_accent_variants", "merge"),
                               ("C04_near_duplicates", "merge")):
            if self._decided(iid) != action_ok:
                continue
            for occ in self.issues_by_id.get(iid, []):
                for col in [c for c in occ.get("columns", []) if c in df.columns]:
                    s = df[col].astype(str).str.strip()
                    repl = {}
                    if iid == "C04_near_duplicates":
                        for p in (occ.get("details") or {}).get("paires", []):
                            repl[p["variante"]] = p["probable"]
                    else:
                        vc = s.value_counts()
                        for g in (occ.get("details") or {}).get("groupes", []):
                            vals = g.get("valeurs", [])
                            if len(vals) > 1:
                                kept = max(vals, key=lambda x: vc.get(x, 0))
                                repl.update({x: kept for x in vals if x != kept})
                    if repl:
                        mask = s.isin(repl)
                        df[col] = s.replace(repl).where(df[col].notna())
                        self._log(iid, "merge_values", rows=int(mask.sum()), cols=[col],
                                  params={"remplacements": len(repl)})
        return df

    # --------------------------------------------------------- M — mesures

    def _measures(self, df: pd.DataFrame) -> pd.DataFrame:
        mesures = [c["name"] for c in self.report.get("_profile_columns", [])
                   if c.get("role_candidate") == "measure"] or \
                  [c for c in (self.mapping.get("revenue"), self.mapping.get("quantity"))
                   if c]

        # Conversion en nombres — INCONDITIONNELLE.
        #
        # Elle ne dépend d'aucune anomalie détectée : sans elle, aucun calcul
        # n'est possible. Lier une opération indispensable à une détection
        # optionnelle laissait les montants en texte quand M04 n'était pas levée.
        converties = []
        for col in mesures:
            # pandas 3.0 utilise le dtype `str`, les versions antérieures `object`.
            # Tester `== object` échouait silencieusement sur Colab.
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(clean_numeric_strings(df[col]), errors="coerce")
                converties.append(col)
        if converties:
            self._log("M04_currency_symbols", "to_numeric", cols=converties,
                      note="Montants convertis en nombres pour permettre les calculs")

        # M03 — valeurs extrêmes
        issue = self.issues.get("M03_extreme_outliers")
        if issue and self._decided("M03_extreme_outliers") == "exclude":
            idx = [i for i in issue.get("sample_rows", []) if i in df.index]
            if idx:
                df = df.drop(index=idx)
                self._log("M03_extreme_outliers", "remove_rows", rows=len(idx),
                          cols=issue["columns"], removed=idx)

        # M01 — montants négatifs
        issue = self.issues.get("M01_negative")
        act = self._decided("M01_negative")
        if issue and act in ("exclude", "separate"):
            for col in [c for c in issue["columns"] if c in df.columns]:
                v = pd.to_numeric(df[col], errors="coerce")
                mask = v < 0
                if mask.any():
                    removed = df.index[mask].tolist()
                    if act == "separate":
                        self.refunds = df[mask].copy()
                    df = df[~mask]
                    self._log("M01_negative",
                              "separate_rows" if act == "separate" else "remove_rows",
                              rows=len(removed), cols=[col], removed=removed)

        # M02 — montants à zéro
        issue = self.issues.get("M02_zero_values")
        if issue and self._decided("M02_zero_values") == "exclude":
            for col in [c for c in issue["columns"] if c in df.columns]:
                v = pd.to_numeric(df[col], errors="coerce")
                mask = v == 0
                if mask.any():
                    removed = df.index[mask].tolist()
                    df = df[~mask]
                    self._log("M02_zero_values", "remove_rows", rows=len(removed),
                              cols=[col], removed=removed)
        return df

    # ----------------------------------------------------------- T — dates

    def _dates(self, df: pd.DataFrame) -> pd.DataFrame:
        date_col = self.mapping.get("order_date")
        if not date_col or date_col not in df.columns:
            return df

        dayfirst = self.decisions.get("T02_ambiguous_format", "dayfirst") == "dayfirst"

        # Le PROFILEUR a déjà déterminé le format en testant chaque motif
        # sur toute la colonne. Le redeviner ici, c'est risquer d'aboutir à
        # une autre conclusion — et une date mal lue décale toute la série
        # temporelle sans qu'aucune erreur ne soit levée.
        fmt = None
        for c in self.report.get("_profile_columns", []):
            if c.get("name") == date_col and c.get("date_format"):
                fmt = c["date_format"]
                break

        if fmt and fmt not in ("mixte", "natif"):
            df[date_col] = pd.to_datetime(df[date_col], format=fmt, errors="coerce")
            self._log("T02_ambiguous_format", "parse_dates", cols=[date_col],
                      params={"format": fmt, "source": "profileur"})
        else:
            # Ordre déduit du CONTENU : une colonne « 04-30-22 » est
            # américaine, « 04/10/2023 » est française. Imposer une convention
            # corrige l'une en cassant l'autre.
            from .profiler import detecter_ordre_jour_mois
            ordre, conf = detecter_ordre_jour_mois(df[date_col].dropna())
            if conf >= 0.7:
                dayfirst = (ordre == "dayfirst")
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce",
                                          format="mixed", dayfirst=dayfirst)
            self._log("T02_ambiguous_format", "parse_dates", cols=[date_col],
                      params={"dayfirst": dayfirst, "ordre_deduit": ordre})

        # T04 — dates absurdes : passage à vide, jamais d'invention de valeur
        mask = (df[date_col].dt.year < 1990) | (df[date_col].dt.year == 1900)
        if mask.any():
            df.loc[mask, date_col] = pd.NaT
            self._log("T04_absurd_dates", "to_null", rows=int(mask.sum()), cols=[date_col])

        # T03 — dates futures
        if self._decided("T03_future_dates") == "exclude":
            mask = df[date_col] > pd.Timestamp.now()
            if mask.any():
                removed = df.index[mask].tolist()
                df = df[~mask]
                self._log("T03_future_dates", "remove_rows", rows=len(removed),
                          cols=[date_col], removed=removed)
        return df

    # ------------------------------------------------------- N — manquants

    def _missing(self, df: pd.DataFrame) -> pd.DataFrame:
        issue = self.issues.get("N01_critical_missing")
        if issue and self._decided("N01_critical_missing") == "exclude":
            for col in [c for c in issue["columns"] if c in df.columns]:
                mask = df[col].isna()
                if mask.any():
                    removed = df.index[mask].tolist()
                    df = df[~mask]
                    self._log("N01_critical_missing", "remove_rows", rows=len(removed),
                              cols=[col], removed=removed,
                              note="Aucune imputation : ligne écartée")
        return df

    # ------------------------------------------------------------ Journal

    def cleaning_log(self) -> dict:
        df_clean = self.result()
        h = hashlib.sha256(
            pd.util.hash_pandas_object(self.df_raw, index=True).values.tobytes()
        ).hexdigest()[:16]

        return {
            "source_hash": f"sha256:{h}",
            "rows_before": int(len(self.df_raw)),
            "rows_after": int(len(df_clean)),
            "rows_removed": int(len(self.df_raw) - len(df_clean)),
            "columns_before": int(self.df_raw.shape[1]),
            "columns_after": int(df_clean.shape[1]),
            "decisions": dict(self.decisions),
            "operations": [o.to_dict() for o in self.operations],
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def summary(self) -> str:
        """Récapitulatif destiné à l'utilisateur, sans jargon."""
        log = self.cleaning_log()
        lignes = [
            f"{log['rows_before']:,} lignes reçues".replace(",", " "),
            f"{log['rows_removed']:,} lignes écartées".replace(",", " "),
            f"{log['rows_after']:,} lignes analysables".replace(",", " "),
        ]
        detail = [f"  • {o['issue_id']:<28} {o['action']:<20} "
                  f"{o['rows_affected'] or '':>6}" for o in log["operations"]]
        return "\n".join(lignes + [""] + detail)


# --------------------------------------------------------------------------
# Contrôle de vraisemblance après nettoyage
# --------------------------------------------------------------------------
#
# Ce qui distingue un analyste d'un script : après avoir corrigé, il regarde
# le résultat et se demande s'il est plausible. Un nettoyage qui fait chuter
# le chiffre d'affaires de 40 % n'est pas forcément faux — mais il doit être
# signalé, pas appliqué en silence.
#
# Le système ne peut pas juger à la place du commerçant. Il peut en revanche
# lui montrer l'ampleur de ce qu'il a fait.

SEUILS_ALERTE = {
    "lignes_retirees_pct": 20.0,
    "revenu_perdu_pct": 15.0,
    "clients_perdus_pct": 20.0,
    "periode_reduite_pct": 10.0,
    "categories_perdues": 1,
}


def controle_vraisemblance(cleaner, mapping: dict | None = None) -> dict:
    """
    Compare les grandeurs métier avant et après nettoyage.

    Retourne un rapport avec le détail des écarts et, le cas échéant, des
    alertes formulées en langage clair. Ce rapport doit être présenté à
    l'utilisateur AVANT qu'il ne valide définitivement.
    """
    m = mapping or cleaner.mapping or {}
    brut, net = cleaner.df_raw, cleaner.result()

    def somme(df, col):
        if not col or col not in df.columns:
            return None
        v = pd.to_numeric(clean_numeric_strings(df[col]), errors="coerce")
        return float(v.sum(skipna=True))

    def distincts(df, col):
        if not col or col not in df.columns:
            return None
        return int(df[col].nunique())

    # Format de date établi par le profileur : on le RÉUTILISE au lieu de
    # relancer une inférence. Comparer une colonne texte reparsée à une colonne
    # déjà typée revenait à comparer deux lectures différentes des mêmes
    # dates — d'où une « réduction de 31,3 % de la période » alors qu'aucune
    # ligne n'avait été retirée.
    fmt_date = None
    for c in (cleaner.report.get("_profile_columns") or []):
        if c.get("role_candidate") == "temporal" and c.get("date_format"):
            if c["date_format"] not in ("mixte", "natif"):
                fmt_date = c["date_format"]
            break

    def amplitude(df, col):
        if not col or col not in df.columns:
            return None
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            d = s.dropna()
        elif fmt_date:
            d = pd.to_datetime(s, format=fmt_date, errors="coerce").dropna()
        else:
            from .profiler import detecter_ordre_jour_mois
            ordre, conf = detecter_ordre_jour_mois(s.dropna())
            d = pd.to_datetime(s, errors="coerce",
                               dayfirst=(ordre != "monthfirst")).dropna()
        if len(d) < 2:
            return None
        return int((d.max() - d.min()).days)

    ecarts, alertes = {}, []

    def comparer(nom, avant, apres, seuil_cle, message):
        if avant in (None, 0) or apres is None:
            return
        variation = (apres - avant) / abs(avant) * 100
        ecarts[nom] = {"avant": round(avant, 2), "apres": round(apres, 2),
                       "variation_pct": round(variation, 2)}
        seuil = SEUILS_ALERTE.get(seuil_cle)
        if seuil is not None and abs(variation) > seuil:
            alertes.append({
                "indicateur": nom,
                "variation_pct": round(variation, 1),
                "message": message.format(pct=abs(round(variation, 1))),
            })

    comparer("lignes", len(brut), len(net), "lignes_retirees_pct",
             "Le nettoyage a retiré {pct} % de vos lignes. "
             "Vérifiez que c'est bien ce que vous vouliez.")

    rev = m.get("revenue")
    comparer("chiffre_affaires", somme(brut, rev), somme(net, rev), "revenu_perdu_pct",
             "Votre chiffre d'affaires baisse de {pct} % après nettoyage. "
             "C'est normal si vous avez retiré les commandes annulées.")

    cust = m.get("customer_id")
    comparer("clients", distincts(brut, cust), distincts(net, cust),
             "clients_perdus_pct",
             "{pct} % de vos clients ont disparu du fichier nettoyé.")

    date = m.get("order_date")
    comparer("periode_jours", amplitude(brut, date), amplitude(net, date),
             "periode_reduite_pct",
             "La période couverte se réduit de {pct} %. "
             "Certains mois pourraient manquer.")

    # Une catégorie entièrement disparue est presque toujours une erreur :
    # le nettoyage ne devrait jamais faire perdre une gamme de produits.
    cat = m.get("product_category")
    if cat and cat in brut.columns and cat in net.columns:
        avant = set(brut[cat].dropna().astype(str).str.strip().str.lower())
        apres = set(net[cat].dropna().astype(str).str.strip().str.lower())
        perdues = avant - apres
        ecarts["categories"] = {"avant": len(avant), "apres": len(apres),
                                "disparues": sorted(perdues)[:10]}
        if len(perdues) >= SEUILS_ALERTE["categories_perdues"]:
            alertes.append({
                "indicateur": "categories",
                "variation_pct": None,
                "message": f"{len(perdues)} catégorie(s) ont totalement disparu : "
                           f"{', '.join(sorted(perdues)[:3])}. "
                           f"Vérifiez que ces produits ne vous intéressent plus.",
            })

    # Le résumé doit refléter les écarts RÉELS, pas seulement l'absence
    # d'alerte. Une première version annonçait « le nettoyage n'a pas modifié
    # vos grandeurs principales » alors que le chiffre d'affaires baissait de
    # 13,7 % — sous le seuil, donc sans alerte, mais loin d'être négligeable.
    # Un résumé qui minimise ce que l'utilisateur peut vérifier lui-même
    # détruit la confiance dans tout le reste.
    notables = [(nom, e["variation_pct"]) for nom, e in ecarts.items()
                if "variation_pct" in e and abs(e["variation_pct"]) >= 2.0]

    if alertes:
        statut = "attention"
        resume = (f"{len(alertes)} point(s) méritent votre attention "
                  f"avant de valider.")
    elif notables:
        statut = "modifie"
        pire = max(notables, key=lambda x: abs(x[1]))
        libelles = {"lignes": "nombre de lignes",
                    "chiffre_affaires": "chiffre d'affaires",
                    "clients": "nombre de clients",
                    "periode_jours": "période couverte"}
        resume = (f"Le nettoyage a modifié vos chiffres : "
                  f"{libelles.get(pire[0], pire[0])} "
                  f"{'−' if pire[1] < 0 else '+'}{abs(pire[1]):.1f} %. "
                  f"Vérifiez le détail ci-dessous.")
    else:
        statut = "ok"
        resume = "Le nettoyage n'a pas modifié vos grandeurs principales."

    return {
        "statut": statut,
        "ecarts": ecarts,
        "alertes": alertes,
        "ecarts_notables": [{"indicateur": n, "variation_pct": v} for n, v in notables],
        "resume": resume,
    }
