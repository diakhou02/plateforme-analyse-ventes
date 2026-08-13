"""
Lecture robuste de fichiers de vente déposés par un utilisateur non technique.

Étape [1] du pipeline. Ne nettoie rien : se contente d'ouvrir correctement
le fichier et de journaliser tout ce qui a dû être deviné.

Anomalies traitées ici : S01 (encodage), S02 (séparateur), S04 (décalage
d'en-tête), S09 (en-têtes dupliqués).
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Encodages testés dans l'ordre. cp1252 et latin-1 couvrent la quasi-totalité
# des exports Excel francophones mal encodés.
ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]

SEPARATORS = [",", ";", "\t", "|"]

# Signature d'un texte lu avec le mauvais encodage (mojibake).
MOJIBAKE = re.compile(r"Ã[©¨ªàâ¯«»]|â€™|â€œ|Ã‰|Ã§")

# Une ligne d'en-tête plausible : peu de cellules vides, peu de nombres purs.
UNNAMED = re.compile(r"^Unnamed:\s*\d+$")


@dataclass
class ReadReport:
    """Tout ce que la lecture a dû deviner. Alimente quality_report.json."""

    filename: str = ""
    extension: str = ""
    encoding: str | None = None
    separator: str | None = None
    header_row: int = 0
    n_rows: int = 0
    n_columns: int = 0
    issues: list[dict] = field(default_factory=list)

    def add(self, issue_id: str, message: str, **details) -> None:
        self.issues.append({"id": issue_id, "message": message, **details})

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "extension": self.extension,
            "encoding_detected": self.encoding,
            "separator_detected": self.separator,
            "header_row": self.header_row,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "issues": self.issues,
        }


def _decode(raw: bytes, report: ReadReport) -> str:
    """Trouve l'encodage. Rejette celui qui produit du mojibake."""
    fallback = None
    for enc in ENCODINGS:
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        if MOJIBAKE.search(text[:20000]):
            # Décodé sans erreur mais visiblement faux : on garde en secours.
            fallback = fallback or (enc, text)
            continue
        report.encoding = enc
        if enc != "utf-8":
            report.add(
                "S01_encoding",
                f"Fichier lu en {enc} et converti en UTF-8.",
                policy="AUTO",
            )
        return text

    if fallback:
        enc, text = fallback
        report.encoding = enc
        report.add(
            "S01_encoding",
            "Des caractères accentués semblent abîmés dans le fichier d'origine.",
            policy="AUTO_NOTIFIED",
        )
        return text

    report.encoding = "utf-8"
    report.add("S01_encoding", "Encodage indéterminé, lecture tolérante.", policy="AUTO")
    return raw.decode("utf-8", errors="replace")


def _sniff_separator(text: str, report: ReadReport) -> str:
    """Séparateur le plus régulier sur les 20 premières lignes non vides."""
    lines = [ln for ln in text.splitlines()[:20] if ln.strip()]
    if not lines:
        return ","

    try:
        sep = csv.Sniffer().sniff("\n".join(lines[:5]), delimiters="".join(SEPARATORS)).delimiter
        report.separator = sep
        return sep
    except csv.Error:
        pass

    # Repli : le séparateur dont le nombre d'occurrences varie le moins.
    best, best_score = ",", (-1, 1e9)
    for sep in SEPARATORS:
        counts = [ln.count(sep) for ln in lines]
        if min(counts) == 0:
            continue
        spread = max(counts) - min(counts)
        score = (min(counts), -spread)
        if score > (best_score[0], -best_score[1]):
            best, best_score = sep, (min(counts), spread)

    report.separator = best
    if best != ",":
        report.add("S02_separator", f"Colonnes séparées par « {best} ».", policy="AUTO")
    return best


def _looks_like_header(cells: list[str]) -> bool:
    """Une ligne d'en-tête : majoritairement du texte non numérique et non vide."""
    cells = [str(c).strip() for c in cells]
    if not cells:
        return False
    filled = [c for c in cells if c and c.lower() not in {"nan", "none"}]
    if len(filled) < max(2, len(cells) * 0.6):
        return False
    numeric = sum(1 for c in filled if re.fullmatch(r"-?[\d\s.,]+", c))
    return numeric <= len(filled) * 0.3


def _find_header_row(text: str, sep: str, report: ReadReport, max_scan: int = 10) -> int:
    """Détecte un en-tête décalé (titre, logo, lignes vides avant le tableau)."""
    rows = list(csv.reader(io.StringIO(text), delimiter=sep))
    for i, row in enumerate(rows[:max_scan]):
        if _looks_like_header(row):
            if i > 0:
                report.add(
                    "S04_header_offset",
                    f"Les vrais noms de colonnes commençaient à la ligne {i + 1}.",
                    policy="AUTO_NOTIFIED",
                    skipped_rows=i,
                )
            return i
    return 0


def _dedupe_columns(df: pd.DataFrame, report: ReadReport) -> pd.DataFrame:
    """S09 : suffixe les noms de colonnes en double."""
    seen: dict[str, int] = {}
    new, renamed = [], []
    for col in df.columns:
        name = str(col).strip()
        if UNNAMED.match(name) or not name:
            name = f"colonne_{len(new) + 1}"
        if name in seen:
            seen[name] += 1
            renamed.append(name)
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        new.append(name)
    df.columns = new
    if renamed:
        report.add(
            "S09_duplicate_headers",
            f"{len(renamed)} nom(s) de colonne en double ont été distingués.",
            policy="AUTO",
            columns=renamed,
        )
    return df


def read_any(source: str | Path | bytes, filename: str | None = None) -> tuple[pd.DataFrame, ReadReport]:
    """
    Ouvre un CSV, TSV ou Excel déposé par l'utilisateur.

    Retourne le DataFrame brut (aucun nettoyage de contenu) et le rapport
    de lecture. Lève ValueError si le fichier est illisible (B04).
    """
    report = ReadReport()

    if isinstance(source, (str, Path)):
        path = Path(source)
        report.filename = filename or path.name
        raw = path.read_bytes()
    else:
        raw = source
        report.filename = filename or "fichier"

    report.extension = Path(report.filename).suffix.lower().lstrip(".")

    if report.extension in {"xlsx", "xls", "xlsm"}:
        df = _read_excel(raw, report)
    else:
        df = _read_text(raw, report)

    df = _dedupe_columns(df, report)
    report.n_rows, report.n_columns = df.shape

    if df.empty or report.n_columns == 0:
        raise ValueError("B04_file_unreadable")

    return df, report


def _read_text(raw: bytes, report: ReadReport) -> pd.DataFrame:
    text = _decode(raw, report)
    sep = _sniff_separator(text, report)
    header = _find_header_row(text, sep, report)
    report.header_row = header

    return pd.read_csv(
        io.StringIO(text),
        sep=sep,
        skiprows=header,
        dtype=str,          # tout en texte : le cast est du ressort du nettoyage
        keep_default_na=False,
        na_values=[""],
        engine="python",
        on_bad_lines="skip",
    )


def _read_excel(raw: bytes, report: ReadReport) -> pd.DataFrame:
    probe = pd.read_excel(io.BytesIO(raw), header=None, nrows=10, dtype=str)
    header = 0
    for i in range(len(probe)):
        if _looks_like_header(probe.iloc[i].fillna("").tolist()):
            header = i
            break
    if header > 0:
        report.add(
            "S04_header_offset",
            f"Les vrais noms de colonnes commençaient à la ligne {header + 1}.",
            policy="AUTO_NOTIFIED",
            skipped_rows=header,
        )
    report.header_row = header
    report.encoding = "n/a (Excel)"
    report.separator = "n/a (Excel)"
    return pd.read_excel(io.BytesIO(raw), header=header, dtype=str)
