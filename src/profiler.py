"""
Profilage structurel d'un jeu de données de vente.

Étape [2] du pipeline. Produit `profile.json` : la seule chose que verra
le planificateur LLM. Les données brutes ne quittent jamais la machine.

Rôles pré-qualifiés ici (règles §1 de la spécification) :
identifier, temporal, measure, categorical, high_cardinality_cat, boolean_flag.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Indices lexicaux. Complètent les heuristiques statistiques, ne les remplacent
# pas : un nom de colonne ne suffit jamais à décider seul.
# --------------------------------------------------------------------------

ID_HINTS = {"id", "ident", "code", "ref", "reference", "sku", "uuid", "num",
            "numero", "no", "identifiant", "key"}

DATE_HINTS = {"date", "jour", "day", "time", "timestamp", "created", "purchase",
              "commande", "achat", "livraison", "shipping", "delivery", "at"}

MEASURE_HINTS = {"prix", "price", "montant", "amount", "total", "ca", "revenue",
                 "chiffre", "cout", "cost", "valeur", "value", "quantite",
                 "quantity", "qty", "qte", "nombre", "freight", "payment",
                 "subtotal", "tarif", "somme"}

# Mesures non sommables : on ne somme jamais un ratio ni une note.
NON_ADDITIVE_HINTS = {"taux", "rate", "ratio", "pct", "pourcentage", "percent",
                      "moyenne", "mean", "avg", "note", "rating", "score",
                      "age", "poids_unitaire", "unit_price", "prix_unitaire",
                      "latitude", "longitude", "lat", "lng", "lon", "zip",
                      "postal", "cp"}

NULL_TOKENS = {"", "na", "n/a", "nan", "none", "null", "-", "--", "?", "#n/a",
               "#na", "inconnu", "non renseigne", "vide", "nil"}

BOOL_TOKENS = {"oui", "non", "yes", "no", "true", "false", "vrai", "faux",
               "1", "0", "y", "n", "o"}

# Tous les espaces, y compris insécable (U+00A0) et fin insécable (U+202F),
# écrits littéralement pour rester compatibles avec le moteur RE2 d'Arrow.
ESPACES = "[" + " \t\r\n" + "  " + "]"

CURRENCY = re.compile(r"[€$£¥]|R\$|EUR|USD|XOF|FCFA|CFA", re.IGNORECASE)
NUM_LIKE = re.compile(r"^-?\d{1,3}(?:[ \u00a0.,]\d{3})*(?:[.,]\d+)?$|^-?\d+(?:[.,]\d+)?$")

DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y",
                "%d.%m.%Y", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M",
                "%Y-%m-%dT%H:%M:%S", "%d %B %Y", "%b %d, %Y"]

SAMPLE_CAP = 20000   # échantillon pour l'inférence de type (§7 passage à l'échelle)


def normalize_name(name: str) -> str:
    """« Prix Unitaire (€) » -> « prix_unitaire »."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return re.sub(r"\s+", "_", s.strip())


def _tokens(name: str) -> set[str]:
    return set(normalize_name(name).split("_"))


def _hits(name: str, hints: set[str]) -> bool:
    toks = _tokens(name)
    if toks & hints:
        return True
    norm = normalize_name(name)
    return any(h in norm for h in hints if len(h) > 3)


# --------------------------------------------------------------------------
# Inférence de type
# --------------------------------------------------------------------------

def comma_role(s: pd.Series) -> tuple[str, float]:
    """
    Détermine si la virgule est un séparateur DÉCIMAL ou de MILLIERS,
    pour la colonne entière.

    Une décision valeur par valeur est impossible : « 0,512 » peut être lu
    0.512 (français) ou 512 (américain). Seul le contexte de la colonne tranche.

    Retourne (rôle, confiance).
    """
    t = s.astype(str).str.strip().str.replace(CURRENCY, "", regex=True)
    # Espaces insécables en clair : le moteur regex d'Arrow (pandas 3.0)
    # rejette les échappements \u, contrairement au module `re` de Python.
    t = t.str.replace(ESPACES, "", regex=True)
    n = len(t)
    if n == 0:
        return "decimal", 0.0

    # Preuves DÉCIMALES
    #  - virgule suivie d'un nombre de chiffres != 3 : « 89,01 », « 1,5 »
    #  - partie entière nulle : « 0,512 » ne s'écrit jamais ainsi en milliers
    dec = int(t.str.contains(r",\d{1,2}$|,\d{4,}$", regex=True, na=False).sum())
    dec += int(t.str.match(r"^-?0,\d+$", na=False).sum())

    # Preuves MILLIERS
    #  - au moins deux groupes : « 1,234,567 »
    #  - une virgule ET un point décimal : « 1,234.56 »
    thou = int(t.str.match(r"^-?\d{1,3}(,\d{3}){2,}$", na=False).sum())
    thou += int(t.str.contains(r",\d{3}\.\d+", regex=True, na=False).sum())

    if dec == 0 and thou == 0:
        return "decimal", 0.5          # aucune preuve : défaut francophone
    if dec >= thou:
        return "decimal", round(dec / max(dec + thou, 1), 2)
    return "thousands", round(thou / (dec + thou), 2)


def clean_numeric_strings(s: pd.Series) -> pd.Series:
    """Retire symboles monétaires et séparateurs de milliers, gère la virgule décimale."""
    t = s.astype(str).str.strip()
    t = t.str.replace(CURRENCY, "", regex=True)
    # Espaces insécables en clair : le moteur regex d'Arrow (pandas 3.0)
    # rejette les échappements \u, contrairement au module `re` de Python.
    t = t.str.replace(ESPACES, "", regex=True)

    role, _ = comma_role(s)

    if role == "thousands":
        # « 1,234.56 » -> « 1234.56 »
        t = t.str.replace(",", "", regex=False)
    else:
        # Format français : « 1.234,56 » -> « 1234.56 », puis « 89,01 » -> « 89.01 »
        fr = t.str.match(r"^-?\d{1,3}(\.\d{3})+,\d+$", na=False)
        t = t.mask(fr, t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
        t = t.str.replace(",", ".", regex=False)

    return t


def _non_null(s: pd.Series) -> pd.Series:
    """Écarte les vrais nuls et les marqueurs textuels de nullité (C05)."""
    t = s.astype(str).str.strip()
    return s[~t.str.lower().isin(NULL_TOKENS) & s.notna()]


def try_numeric(s: pd.Series) -> tuple[pd.Series | None, float]:
    if pd.api.types.is_numeric_dtype(s):
        return s, 1.0
    vals = _non_null(s)
    if vals.empty:
        return None, 0.0
    parsed = pd.to_numeric(clean_numeric_strings(vals), errors="coerce")
    ratio = float(parsed.notna().mean())
    return (parsed if ratio >= 0.90 else None), ratio


def try_datetime(s: pd.Series) -> tuple[pd.Series | None, float, str | None]:
    # Déjà typée : rien à inférer.
    if pd.api.types.is_datetime64_any_dtype(s):
        return s, 1.0, "natif"

    # Une colonne NUMÉRIQUE n'est jamais une date. Sans ce garde-fou,
    # pd.to_datetime(46.62) réussit — pandas lit le nombre comme un timestamp
    # epoch — et tous les montants se retrouvent classés en dates dès qu'on
    # reprofile un fichier déjà nettoyé.
    if pd.api.types.is_numeric_dtype(s):
        return None, 0.0, None

    vals = _non_null(s)
    if vals.empty:
        return None, 0.0, None

    # Un identifiant purement numérique ne doit jamais être pris pour une date.
    if vals.astype(str).str.fullmatch(r"\d{5,}").mean() > 0.5:
        return None, 0.0, None

    # Normalisation en str Python : numpy.str_ (produit par np.random.choice,
    # les lectures Parquet, etc.) fait échouer pd.to_datetime avec `format`.
    # Le profileur teste CHAQUE colonne texte comme date potentielle : il ne
    # doit jamais planter sur une colonne de libellés ordinaire.
    vals = vals.map(lambda x: str(x))

    best, best_ratio, best_fmt = None, 0.0, None
    for fmt in DATE_FORMATS:
        try:
            parsed = pd.to_datetime(vals, format=fmt, errors="coerce")
        except Exception:
            continue
        ratio = float(parsed.notna().mean())
        if ratio > best_ratio:
            best, best_ratio, best_fmt = parsed, ratio, fmt
        if ratio >= 0.99:
            break

    if best_ratio < 0.90:  # dernier recours : parsing tolérant (T05)
        try:
            parsed = pd.to_datetime(vals, errors="coerce", format="mixed", dayfirst=True)
            ratio = float(parsed.notna().mean())
            if ratio > best_ratio:
                best, best_ratio, best_fmt = parsed, ratio, "mixte"
        except Exception:
            pass

    return (best if best_ratio >= 0.90 else None), best_ratio, best_fmt


# --------------------------------------------------------------------------
# Profil d'une colonne
# --------------------------------------------------------------------------

@dataclass
class ColumnProfile:
    name: str
    normalized_name: str
    inferred_type: str                 # numeric | datetime | boolean | text
    role_candidate: str                # cf. §1 de la spécification
    role_confidence: float
    null_rate: float
    n_unique: int
    cardinality_ratio: float
    sample_values: list = field(default_factory=list)
    additive: bool | None = None
    stats: dict = field(default_factory=dict)
    top_values: list = field(default_factory=list)
    date_format: str | None = None
    flags: list = field(default_factory=list)


def profile_column(s: pd.Series, n_rows: int) -> ColumnProfile:
    name = str(s.name)
    sample = s.sample(SAMPLE_CAP, random_state=0) if len(s) > SAMPLE_CAP else s

    vals = _non_null(sample)
    null_rate = 1.0 - (len(vals) / len(sample)) if len(sample) else 1.0
    n_unique = int(vals.nunique())
    card_ratio = n_unique / len(vals) if len(vals) else 0.0

    flags: list[str] = []
    num, num_ratio = try_numeric(sample)
    dt, dt_ratio, dt_fmt = try_datetime(sample)

    # Ambiguïté du séparateur décimal : « 1,250 » = 1.25 ou 1250 ?
    # On ne devine pas silencieusement, on signale pour arbitrage utilisateur (M07).
    if num is not None:
        has_comma = vals.astype(str).str.contains(",", na=False).mean() > 0.5
        role_virgule, conf_virgule = comma_role(vals)
        if has_comma and conf_virgule < 0.7:
            flags.append("M07_ambiguous_decimal")

    # --- type ---
    # Le dtype natif fait foi : reprofiler un fichier déjà nettoyé ne doit pas
    # rejouer une inférence textuelle sur des colonnes déjà converties.
    if pd.api.types.is_datetime64_any_dtype(sample):
        inferred, parsed = "datetime", _non_null(sample)
    elif pd.api.types.is_numeric_dtype(sample):
        inferred, parsed = "numeric", _non_null(sample)
    elif dt is not None and dt_ratio >= num_ratio and _hits(name, DATE_HINTS) | (dt_ratio > 0.95):
        inferred, parsed = "datetime", dt
    elif num is not None:
        inferred, parsed = "numeric", num
    elif n_unique == 2 and set(vals.astype(str).str.lower().unique()) <= BOOL_TOKENS:
        inferred, parsed = "boolean", vals
    else:
        inferred, parsed = "text", vals

    # --- rôle ---
    # Précédence explicite. Corrige deux défauts de la règle brute « cardinalité
    # > 0,9 => identifiant » : elle classait les prix (continus, donc presque
    # tous distincts) en identifiants, et ratait les clés étrangères répétées
    # comme customer_id, pourtant indispensables aux analyses de rétention.
    is_measure_name = _hits(name, MEASURE_HINTS) and not _hits(name, ID_HINTS)
    is_id_name = _hits(name, ID_HINTS) and not _hits(name, NON_ADDITIVE_HINTS)
    has_decimals = bool(
        inferred == "numeric" and parsed is not None
        and (parsed.dropna() % 1 != 0).mean() > 0.05
    )

    conf = 0.6
    if len(vals) == 0:
        role, conf = "empty", 1.0
        flags.append("S05_empty_column")
    elif n_unique == 1:
        role, conf = "constant", 1.0
        flags.append("S06_constant_column")
    elif inferred == "numeric" and is_measure_name:
        # Un nom explicite de mesure prime sur la cardinalité : une quantité
        # ne prend que 4 valeurs distinctes et reste une mesure.
        role, conf = "measure", 0.95
    elif inferred == "datetime":
        # AVANT la règle de cardinalité : une colonne de dates toutes distinctes
        # (une commande par jour) atteint un ratio de 1,0 et était classée
        # « identifiant ». Toutes les analyses temporelles disparaissaient.
        role, conf = "temporal", min(0.99, 0.6 + dt_ratio * 0.4)
    elif is_id_name and inferred != "datetime" and n_unique > 10:
        # Clé étrangère : répétée, donc faible cardinalité, mais bien un identifiant.
        role, conf = "identifier", 0.9 if card_ratio > 0.5 else 0.8
    elif card_ratio > 0.9 and len(vals) > 20 and not has_decimals:
        # Un numérique à décimales n'est jamais un identifiant.
        role, conf = "identifier", 0.85
    elif inferred == "boolean" or n_unique == 2:
        role, conf = "boolean_flag", 0.85
    elif inferred == "numeric" and (card_ratio > 0.05 or has_decimals):
        role, conf = "measure", 0.75
    elif inferred == "numeric":
        role, conf = "categorical", 0.6   # numérique mais très répétitif : un code
        flags.append("numeric_but_categorical")
    elif 2 <= n_unique <= 200:
        role, conf = "categorical", 0.85
    elif n_unique > 200:
        role, conf = "high_cardinality_cat", 0.8
        flags.append("excluded_from_groupby")
    else:
        role, conf = "text", 0.5

    # --- additivité (une mesure non sommable ne doit jamais recevoir agg=sum) ---
    additive = None
    if role == "measure":
        additive = True
        if _hits(name, NON_ADDITIVE_HINTS):
            additive = False
        elif parsed.between(0, 1).mean() > 0.95:   # ressemble à un ratio
            additive = False
            flags.append("looks_like_ratio")

    # --- statistiques ---
    stats: dict = {}
    top: list = []
    if role in {"measure", "categorical"} and inferred == "numeric":
        p = parsed.dropna()
        if len(p):
            stats = {
                "min": float(p.min()), "max": float(p.max()),
                "mean": round(float(p.mean()), 2),
                "median": round(float(p.median()), 2),
                "std": round(float(p.std()), 2) if len(p) > 1 else 0.0,
                "skew": round(float(p.skew()), 2) if len(p) > 2 else 0.0,
                "n_negative": int((p < 0).sum()),
                "n_zero": int((p == 0).sum()),
            }
    elif role == "temporal" and parsed is not None:
        p = parsed.dropna()
        if len(p):
            span = (p.max() - p.min()).days
            stats = {
                "min": str(p.min().date()), "max": str(p.max().date()),
                "span_days": int(span),
                "n_future": int((p > pd.Timestamp.now()).sum()),
            }
            if span < 56:
                flags.append("T06_insufficient_span")

    if role in {"categorical", "boolean_flag", "high_cardinality_cat"}:
        vc = vals.astype(str).value_counts().head(10)
        top = [{"value": k, "count": int(v)} for k, v in vc.items()]

    return ColumnProfile(
        name=name,
        normalized_name=normalize_name(name),
        inferred_type=inferred,
        role_candidate=role,
        role_confidence=round(conf, 2),
        null_rate=round(null_rate, 4),
        n_unique=n_unique,
        cardinality_ratio=round(card_ratio, 4),
        sample_values=[str(v) for v in vals.head(3).tolist()],
        additive=additive,
        stats=stats,
        top_values=top,
        date_format=dt_fmt if role == "temporal" else None,
        flags=flags,
    )


def build_profile(df: pd.DataFrame, read_report=None) -> dict:
    """Construit `profile.json`. Entrée unique du planificateur LLM."""
    cols = [profile_column(df[c], len(df)) for c in df.columns]

    by_role: dict[str, list[str]] = {}
    for c in cols:
        by_role.setdefault(c.role_candidate, []).append(c.name)

    return {
        "dataset": {
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "source_hint": (read_report.filename if read_report else None),
        },
        "read_report": (read_report.to_dict() if read_report else None),
        "columns": [asdict(c) for c in cols],
        "roles_summary": by_role,
    }
