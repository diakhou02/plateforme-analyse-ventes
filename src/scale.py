"""
Passage à l'échelle : pandas, Polars et DuckDB comparés.

Le pipeline a été conçu et validé sur des fichiers de quelques milliers de
lignes. Ce module répond à la question que pose l'intitulé « IA et Big Data » :
que devient l'architecture quand le volume augmente de quatre ordres de grandeur ?

TROIS OPÉRATIONS MESURÉES, correspondant aux étapes réelles du pipeline :
  1. LECTURE     — ouvrir le fichier (reader.py)
  2. PROFILAGE   — types, cardinalités, valeurs manquantes (profiler.py)
  3. AGRÉGATION  — group by mesure × dimension (executor.py)

DEUX STRATÉGIES COMPARÉES :
  - traitement intégral : tout est chargé et parcouru
  - diagnostic sur échantillon, correction sur l'intégralité

La seconde est celle qu'implémente déjà le profileur (SAMPLE_CAP = 20 000) ;
ce module en mesure le gain et le coût en précision.
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import psutil


# --------------------------------------------------------------------------
# Génération de jeux de données à volume contrôlé
# --------------------------------------------------------------------------

CATEGORIES = ["Téléphones", "Accessoires", "Coques", "Chargeurs", "Écrans",
              "Câbles", "Écouteurs", "Batteries", "Supports", "Housses"]
VILLES = ["Dakar", "Thiès", "Saint-Louis", "Kaolack", "Ziguinchor",
          "Touba", "Mbour", "Rufisque"]
STATUTS = ["livré"] * 17 + ["annulé", "remboursé", "en cours"]


def generate_dataset(n_rows: int, path: str | Path, seed: int = 0,
                     chunk: int = 500_000) -> Path:
    """
    Écrit un CSV de `n_rows` lignes, par blocs pour ne pas saturer la mémoire.

    Écrire 10 millions de lignes en une fois demanderait plusieurs gigaoctets
    de RAM — soit exactement le problème qu'on cherche à mesurer. La génération
    par blocs est elle-même une illustration de la contrainte.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    debut = np.datetime64("2022-01-01")

    premier = True
    restant = n_rows
    offset = 0
    while restant > 0:
        k = min(chunk, restant)
        bloc = pd.DataFrame({
            "order_id": [f"C{i:09d}" for i in range(offset, offset + k)],
            "customer_id": [f"CL{i:07d}" for i in rng.integers(0, max(1, n_rows // 3), k)],
            "date": debut + rng.integers(0, 1000, k).astype("timedelta64[D]"),
            "categorie": rng.choice(CATEGORIES, k),
            "ville": rng.choice(VILLES, k),
            "statut": rng.choice(STATUTS, k),
            "quantite": rng.integers(1, 6, k),
            "montant": np.round(rng.lognormal(3.5, 0.7, k), 2),
        })
        bloc.to_csv(path, mode="w" if premier else "a",
                    header=premier, index=False, encoding="utf-8")
        premier = False
        offset += k
        restant -= k
        del bloc
        gc.collect()

    return path


# --------------------------------------------------------------------------
# Mesure
# --------------------------------------------------------------------------

@dataclass
class Mesure:
    moteur: str
    operation: str
    n_rows: int
    duree_s: float
    memoire_mo: float
    resultat: str = ""
    erreur: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _memoire_mo() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def chronometre(fn, moteur: str, operation: str, n_rows: int) -> Mesure:
    """
    Exécute une fonction en mesurant temps et pic mémoire.

    Le `gc.collect()` préalable est indispensable : sans lui, la mémoire du test
    précédent fausse la mesure suivante — l'écart observé entre deux moteurs
    serait en partie un artefact de l'ordre d'exécution.
    """
    gc.collect()
    avant = _memoire_mo()
    t0 = time.perf_counter()
    try:
        res = fn()
        erreur = None
    except MemoryError:
        return Mesure(moteur, operation, n_rows, float("nan"), float("nan"),
                      erreur="mémoire insuffisante")
    except Exception as e:
        return Mesure(moteur, operation, n_rows, float("nan"), float("nan"),
                      erreur=f"{type(e).__name__} : {str(e)[:120]}")
    duree = time.perf_counter() - t0
    pic = max(0.0, _memoire_mo() - avant)
    return Mesure(moteur, operation, n_rows, round(duree, 3), round(pic, 1),
                  resultat=str(res)[:80], erreur=erreur)


# --------------------------------------------------------------------------
# Implémentations par moteur
# --------------------------------------------------------------------------

def bench_pandas(path: Path, n_rows: int) -> list[Mesure]:
    out = []
    conteneur = {}

    def lire():
        conteneur["df"] = pd.read_csv(path, parse_dates=["date"])
        return f"{len(conteneur['df'])} lignes"

    out.append(chronometre(lire, "pandas", "lecture", n_rows))
    if out[-1].erreur:
        return out

    df = conteneur["df"]

    def profiler():
        stats = {}
        for c in df.columns:
            stats[c] = {"n_unique": df[c].nunique(), "nulls": df[c].isna().sum()}
        return f"{len(stats)} colonnes"

    out.append(chronometre(profiler, "pandas", "profilage", n_rows))

    def agreger():
        r = (df.assign(mois=df["date"].dt.to_period("M"))
               .groupby(["mois", "categorie"], observed=True)["montant"].sum())
        return f"{len(r)} groupes"

    out.append(chronometre(agreger, "pandas", "agrégation", n_rows))

    del df, conteneur
    gc.collect()
    return out


def bench_polars(path: Path, n_rows: int) -> list[Mesure]:
    import polars as pl
    out = []
    conteneur = {}

    def lire():
        conteneur["df"] = pl.read_csv(path, try_parse_dates=True)
        return f"{conteneur['df'].height} lignes"

    out.append(chronometre(lire, "polars", "lecture", n_rows))
    if out[-1].erreur:
        return out

    df = conteneur["df"]

    def profiler():
        stats = df.select([
            pl.col(c).n_unique().alias(f"{c}_u") for c in df.columns
        ] + [pl.col(c).null_count().alias(f"{c}_n") for c in df.columns])
        return f"{stats.width} mesures"

    out.append(chronometre(profiler, "polars", "profilage", n_rows))

    def agreger():
        r = (df.with_columns(pl.col("date").dt.truncate("1mo").alias("mois"))
               .group_by(["mois", "categorie"])
               .agg(pl.col("montant").sum()))
        return f"{r.height} groupes"

    out.append(chronometre(agreger, "polars", "agrégation", n_rows))

    del df, conteneur
    gc.collect()
    return out


def bench_duckdb(path: Path, n_rows: int) -> list[Mesure]:
    """
    DuckDB requête le CSV SANS le charger en mémoire.

    C'est une différence de nature, pas de degré : les deux autres moteurs
    doivent d'abord matérialiser le fichier. La « lecture » mesurée ici est
    donc l'ouverture d'une vue, ce qui explique des temps quasi nuls — il faut
    en tenir compte dans l'interprétation.
    """
    import duckdb
    out = []
    con = duckdb.connect()
    p = str(path).replace("'", "''")

    def lire():
        con.execute(f"CREATE OR REPLACE VIEW v AS SELECT * FROM read_csv_auto('{p}')")
        n = con.execute("SELECT count(*) FROM v").fetchone()[0]
        return f"{n} lignes"

    out.append(chronometre(lire, "duckdb", "lecture", n_rows))
    if out[-1].erreur:
        con.close()
        return out

    def profiler():
        cols = [r[0] for r in con.execute("DESCRIBE v").fetchall()]
        expr = ", ".join(
            f'count(distinct "{c}") AS "u_{c}", '
            f'sum(case when "{c}" is null then 1 else 0 end) AS "n_{c}"'
            for c in cols)
        con.execute(f"SELECT {expr} FROM v").fetchone()
        return f"{len(cols)} colonnes"

    out.append(chronometre(profiler, "duckdb", "profilage", n_rows))

    def agreger():
        r = con.execute("""
            SELECT date_trunc('month', date) AS mois, categorie,
                   sum(montant) AS total
            FROM v GROUP BY 1, 2
        """).fetchall()
        return f"{len(r)} groupes"

    out.append(chronometre(agreger, "duckdb", "agrégation", n_rows))
    con.close()
    gc.collect()
    return out


MOTEURS = {"pandas": bench_pandas, "polars": bench_polars, "duckdb": bench_duckdb}


def run_benchmark(volumes: list[int], dossier: str | Path,
                  moteurs: list[str] | None = None,
                  garder_fichiers: bool = False) -> pd.DataFrame:
    """Exécute le benchmark complet et retourne un tableau de mesures."""
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    moteurs = moteurs or list(MOTEURS)
    lignes = []

    for n in volumes:
        chemin = dossier / f"bench_{n}.csv"
        if not chemin.exists():
            t0 = time.perf_counter()
            generate_dataset(n, chemin)
            print(f"  {n:>10,} lignes générées en {time.perf_counter()-t0:.1f} s "
                  f"({chemin.stat().st_size/1e6:.0f} Mo)".replace(",", " "))

        taille = chemin.stat().st_size / 1e6
        for nom in moteurs:
            for m in MOTEURS[nom](chemin, n):
                d = m.to_dict()
                d["taille_mo"] = round(taille, 1)
                lignes.append(d)
                etat = m.erreur or f"{m.duree_s:>7.2f} s  {m.memoire_mo:>7.0f} Mo"
                print(f"    {nom:<8} {m.operation:<12} {etat}")

        if not garder_fichiers:
            chemin.unlink(missing_ok=True)

    return pd.DataFrame(lignes)


# --------------------------------------------------------------------------
# Échantillonnage : le compromis du profileur
# --------------------------------------------------------------------------

def bench_echantillonnage(path: Path, n_rows: int,
                          tailles: list[int]) -> pd.DataFrame:
    """
    Compare le profilage intégral au profilage sur échantillon.

    Mesure le gain de temps ET l'écart d'estimation. Un échantillon qui divise
    le temps par cent mais fausse les cardinalités ne serait pas acceptable :
    le mapping sémantique dépend directement de `n_unique`.
    """
    df = pd.read_csv(path, parse_dates=["date"])

    def profil(d):
        return {c: {"n_unique": int(d[c].nunique()),
                    "null_rate": float(d[c].isna().mean())} for c in d.columns}

    t0 = time.perf_counter()
    ref = profil(df)
    t_ref = time.perf_counter() - t0

    lignes = [{"taille_echantillon": len(df), "duree_s": round(t_ref, 3),
               "acceleration": 1.0, "erreur_cardinalite_pct": 0.0,
               "erreur_nulls_pct": 0.0}]

    for k in tailles:
        if k >= len(df):
            continue
        ech = df.sample(k, random_state=0)
        t0 = time.perf_counter()
        est = profil(ech)
        t = time.perf_counter() - t0

        # L'erreur qui compte n'est pas sur le nombre absolu de valeurs
        # distinctes — un échantillon en verra forcément moins — mais sur la
        # DÉCISION qui en découle : la colonne est-elle un identifiant, une
        # catégorie, une mesure ? C'est le ratio de cardinalité qui tranche.
        err_card, err_null = [], []
        for c in df.columns:
            r_ref = ref[c]["n_unique"] / len(df)
            r_est = est[c]["n_unique"] / k
            err_card.append(abs(r_est - r_ref))
            err_null.append(abs(est[c]["null_rate"] - ref[c]["null_rate"]))

        lignes.append({
            "taille_echantillon": k,
            "duree_s": round(t, 3),
            "acceleration": round(t_ref / t, 1) if t else float("inf"),
            "erreur_cardinalite_pct": round(float(np.mean(err_card)) * 100, 2),
            "erreur_nulls_pct": round(float(np.mean(err_null)) * 100, 3),
        })

    del df
    gc.collect()
    return pd.DataFrame(lignes)


# --------------------------------------------------------------------------
# Profilage hybride : le correctif au biais d'échantillonnage
# --------------------------------------------------------------------------

def cardinalites_exactes(path: Path, moteur: str = "duckdb") -> tuple[dict, float]:
    """
    Compte les valeurs distinctes sur l'INTÉGRALITÉ du fichier.

    Motivation — l'échantillonnage fausse gravement le ratio de cardinalité :
    une colonne `customer_id` à 0,317 sur 1 M de lignes remonte à 0,97 sur un
    échantillon de 20 000, parce qu'un client a peu de chances d'y apparaître
    deux fois. Elle bascule alors de « clé étrangère » à « identifiant », et
    toutes les analyses de fidélité disparaissent silencieusement.

    Le compromis retenu : cardinalités exactes en une passe (opération peu
    coûteuse, surtout en SQL), tout le reste sur échantillon.
    """
    t0 = time.perf_counter()

    if moteur == "duckdb":
        import duckdb
        con = duckdb.connect()
        p = str(path).replace("'", "''")
        con.execute(f"CREATE OR REPLACE VIEW v AS SELECT * FROM read_csv_auto('{p}')")
        cols = [r[0] for r in con.execute("DESCRIBE v").fetchall()]
        expr = ", ".join(f'count(distinct "{c}")' for c in cols)
        n_total = con.execute("SELECT count(*) FROM v").fetchone()[0]
        vals = con.execute(f"SELECT {expr} FROM v").fetchone()
        con.close()
        card = {c: {"n_unique": int(v), "ratio": round(v / n_total, 4)}
                for c, v in zip(cols, vals)}
    else:
        import polars as pl
        lf = pl.scan_csv(path)
        n_total = lf.select(pl.len()).collect().item()
        cols = lf.collect_schema().names()
        res = lf.select([pl.col(c).n_unique().alias(c) for c in cols]).collect()
        card = {c: {"n_unique": int(res[c][0]),
                    "ratio": round(res[c][0] / n_total, 4)} for c in cols}

    return card, round(time.perf_counter() - t0, 3)


def comparer_strategies(path: Path, taille_ech: int = 20_000) -> pd.DataFrame:
    """
    Compare trois stratégies de profilage, COÛT DE LECTURE INCLUS.

    Inclure la lecture est indispensable à l'honnêteté de la comparaison :
    profiler un DataFrame déjà en mémoire ne dit rien du cas réel, où le
    fichier doit d'abord être ouvert. C'est précisément ce chargement que
    l'échantillonnage permet d'éviter — et c'est là qu'est le gain.

      1. intégral    — lecture complète + profilage complet
      2. échantillon — lecture partielle + profilage (rapide, mais fausse les rôles)
      3. hybride     — cardinalités exactes en SQL (sans chargement)
                       + échantillon chargé pour le reste
    """
    def role(n_unique: int, ratio: float) -> str:
        if ratio > 0.9:
            return "identifier"
        if n_unique <= 200:
            return "categorical"
        return "high_cardinality"

    # --- 1. Intégral : lecture + profilage ---
    gc.collect()
    m0 = _memoire_mo()
    t0 = time.perf_counter()
    df = pd.read_csv(path)
    n = len(df)
    ref = {c: (int(df[c].nunique()), df[c].nunique() / n) for c in df.columns}
    t_integral = time.perf_counter() - t0
    mem_integral = _memoire_mo() - m0
    roles_ref = {c: role(*v) for c, v in ref.items()}
    colonnes = list(df.columns)
    del df
    gc.collect()

    # --- 2. Échantillon seul : on ne lit que les premières lignes ---
    m0 = _memoire_mo()
    t0 = time.perf_counter()
    ech = pd.read_csv(path, nrows=taille_ech)
    est = {c: (int(ech[c].nunique()), ech[c].nunique() / len(ech)) for c in ech.columns}
    t_ech = time.perf_counter() - t0
    mem_ech = _memoire_mo() - m0
    roles_ech = {c: role(*v) for c, v in est.items()}
    del ech
    gc.collect()

    # --- 3. Hybride : cardinalités en SQL (sans charger) + échantillon ---
    gc.collect()
    m0 = _memoire_mo()
    t0 = time.perf_counter()
    card, _ = cardinalites_exactes(path)
    ech = pd.read_csv(path, nrows=taille_ech)
    t_hybride = time.perf_counter() - t0
    mem_hybride = _memoire_mo() - m0
    roles_hyb = {c: role(card[c]["n_unique"], card[c]["ratio"]) for c in colonnes}
    del ech
    gc.collect()

    def justes(roles):
        return sum(1 for c in roles_ref if roles.get(c) == roles_ref[c])

    total = len(roles_ref)
    faux = [c for c in roles_ref if roles_ech.get(c) != roles_ref[c]]

    return pd.DataFrame([
        {"strategie": "intégral", "duree_s": round(t_integral, 2),
         "memoire_mo": round(mem_integral), "roles_corrects": f"{total}/{total}",
         "exactitude": 1.0, "acceleration": 1.0, "roles_faux": ""},
        {"strategie": f"échantillon {taille_ech}", "duree_s": round(t_ech, 2),
         "memoire_mo": round(mem_ech), "roles_corrects": f"{justes(roles_ech)}/{total}",
         "exactitude": round(justes(roles_ech) / total, 3),
         "acceleration": round(t_integral / t_ech, 1) if t_ech else None,
         "roles_faux": ", ".join(faux)},
        {"strategie": "hybride", "duree_s": round(t_hybride, 2),
         "memoire_mo": round(mem_hybride), "roles_corrects": f"{justes(roles_hyb)}/{total}",
         "exactitude": round(justes(roles_hyb) / total, 3),
         "acceleration": round(t_integral / t_hybride, 1) if t_hybride else None,
         "roles_faux": ""},
    ])
