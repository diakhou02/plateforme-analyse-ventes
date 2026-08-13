"""
Planificateur : le LLM décide QUOI analyser.

Étape [5] du pipeline. Produit `plan.json` — une liste de spécifications
d'analyse, jamais de chiffres.

PRINCIPE FONDAMENTAL
    Le LLM décide quoi analyser. Le code calcule combien.

Le modèle ne reçoit que le PROFIL du jeu de données (noms de colonnes, types,
plages de valeurs), jamais les données elles-mêmes. Cela protège les données du
commerçant, réduit le coût en tokens, et rend impossible qu'un chiffre affiché
provienne du modèle plutôt que d'un calcul.

Le plan produit est ensuite validé (validator.py) puis exécuté (executor.py).
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .mapper import CONCEPTS

# --------------------------------------------------------------------------
# Grammaire des specs — énumérations strictes
# --------------------------------------------------------------------------

AGGREGATIONS = {"sum", "mean", "median", "count", "count_distinct", "min", "max", "std"}
DIM_TYPES = {"temporal", "categorical", "numeric_binned"}
GRAINS = {"day", "week", "month", "quarter", "year", "dayofweek", "hour"}
FILTER_OPS = {"eq", "neq", "in", "not_in", "gt", "gte", "lt", "lte", "between", "not_null"}
CHARTS = {"line", "bar", "grouped_bar", "stacked_bar", "area", "scatter",
          "heatmap", "histogram", "boxplot", "treemap"}
PATTERN_HINTS = {
    "trend", "trend_decline", "seasonality", "anomaly", "anomaly_spike",
    "pareto_concentration", "long_tail", "long_tail_dead_stock", "basket_erosion",
    "weak_retention", "single_purchase_dominance", "day_of_week_peak",
    "seasonal_peak", "geographic_concentration", "high_cancellation",
    "divergence_with_volume",
}


# --------------------------------------------------------------------------
# Profil condensé transmis au modèle
# --------------------------------------------------------------------------

def build_llm_profile(profile: dict, mapping_result: dict,
                      quality_report: dict | None = None) -> dict:
    """
    Condensé du profil, calibré pour le LLM.

    On retire tout ce qui ne sert pas à décider QUOI analyser : échantillons de
    valeurs superflus, statistiques de second ordre, métadonnées de lecture.
    Un profil de 8 Ko descend à environ 2 Ko — moins de tokens, et surtout moins
    de bruit dans lequel le modèle pourrait se perdre.
    """
    mapping = mapping_result.get("mapping", {})
    inverse = {v: k for k, v in mapping.items() if v}

    cols = []
    for c in profile["columns"]:
        if c["role_candidate"] in {"empty", "constant"}:
            continue
        entry = {
            "name": c["name"],
            "role": c["role_candidate"],
            "n_unique": c["n_unique"],
        }
        if c["name"] in inverse:
            entry["business_role"] = inverse[c["name"]]
        if c["role_candidate"] == "measure":
            entry["additive"] = c["additive"]
            st = c["stats"]
            if st:
                entry["range"] = [st.get("min"), st.get("max")]
                entry["median"] = st.get("median")
        elif c["role_candidate"] == "temporal":
            st = c["stats"]
            if st:
                entry["period"] = [st.get("min"), st.get("max")]
                entry["span_days"] = st.get("span_days")
        elif c["role_candidate"] in {"categorical", "boolean_flag"}:
            entry["values"] = [v["value"] for v in c.get("top_values", [])[:6]]
        if c["null_rate"] > 0.05:
            entry["null_rate"] = round(c["null_rate"], 2)
        cols.append(entry)

    dispo = mapping_result.get("available_analyses", {})
    return {
        "dataset": {
            "n_rows": profile["dataset"]["n_rows"],
            "n_columns": len(cols),
        },
        "columns": cols,
        "business_mapping": {k: v for k, v in mapping.items() if v},
        "available_analyses": dispo.get("available", []),
        "unavailable_analyses": [
            {"analysis": a["analysis"], "reason": f"colonne manquante : {', '.join(a['missing'])}"}
            for a in dispo.get("unavailable", [])
        ],
        "data_quality_note": (
            f"Données nettoyées, score de qualité {quality_report['quality_score']}/100"
            if quality_report else "Données nettoyées"
        ),
    }


# --------------------------------------------------------------------------
# Prompt système
# --------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """\
Tu es un data analyst senior spécialisé dans le e-commerce. On te confie le
profil structurel d'un jeu de données appartenant à un commerçant en ligne.

Ta mission : concevoir le plan d'analyse qui répondra à sa question implicite —
« qu'est-ce qui marche et qu'est-ce qui ne marche pas dans mon commerce ? »

TU NE VOIS PAS LES DONNÉES. Tu ne vois que leur structure. Tu ne dois donc
JAMAIS produire, estimer ou deviner une valeur chiffrée issue des données.
Tu produis uniquement des SPÉCIFICATIONS D'ANALYSE. Un moteur de calcul
séparé les exécutera.

RÈGLES DE PRODUCTION
1. Produis entre 15 et 20 spécifications, classées par priorité décroissante
   (priority = 1 pour la plus importante).
2. Chaque spécification comporte exactement 1 mesure et 1 ou 2 dimensions.
3. N'utilise QUE les noms de colonnes présents dans le profil, à l'identique,
   caractère pour caractère, accents et espaces compris.
4. N'utilise l'agrégation "sum" que sur les colonnes marquées additive: true.
5. N'agrège jamais une colonne de rôle "identifier", sauf en count_distinct.
6. Si la dimension principale a plus de 20 valeurs distinctes, renseigne
   obligatoirement "limit".
7. N'utilise le graphique "line" que si une dimension est temporelle.
8. Ignore les analyses listées dans "unavailable_analyses".

CRITÈRES DE PERTINENCE — priorise dans cet ordre
a. Ce qui révèle une TENDANCE (l'activité monte-t-elle ou descend-elle ?)
b. Ce qui révèle une CONCENTRATION ou un DÉSÉQUILIBRE (dépendance, angle mort)
c. Ce qui révèle un COMPORTEMENT CLIENT (fidélité, réachat, valeur)
d. Ce qui révèle une OPPORTUNITÉ NON EXPLOITÉE (créneau, région, segment)
e. Les croisements à 2 dimensions qui expliquent un phénomène observé en (a)

Évite : les analyses purement descriptives sans enjeu de décision, les
redondances, et les croisements dont aucune action commerciale ne pourrait
découler.

Pour chaque spécification, "business_rationale" doit expliquer quelle DÉCISION
le commerçant pourra prendre à la lecture du résultat. Si tu ne peux pas
formuler cette décision, ne propose pas la spécification.

VALEURS AUTORISÉES
  agg    : sum, mean, median, count, count_distinct, min, max, std
  type   : temporal, categorical, numeric_binned
  grain  : day, week, month, quarter, year, dayofweek, hour
  op     : eq, neq, in, not_in, gt, gte, lt, lte, between, not_null
  chart  : line, bar, grouped_bar, stacked_bar, area, scatter, heatmap,
           histogram, boxplot, treemap

FORMAT DE SORTIE
Réponds UNIQUEMENT par un objet JSON valide de cette forme, sans texte avant
ni après, sans bloc de code Markdown :

{
  "dataset_understanding": "une ou deux phrases sur ce que fait ce commerce",
  "specs": [
    {
      "id": "spec_001",
      "priority": 1,
      "title": "titre court en français",
      "business_question": "la question que se pose le commerçant",
      "measure": {"column": "...", "agg": "sum", "alias": "..."},
      "dimensions": [{"column": "...", "type": "temporal", "grain": "month"}],
      "filters": [],
      "sort": {"by": "dimension", "order": "asc"},
      "limit": null,
      "chart": "line",
      "business_rationale": "quelle décision cette analyse permet de prendre",
      "pattern_hints": ["trend", "seasonality"]
    }
  ]
}"""


def build_user_prompt(llm_profile: dict) -> str:
    return (
        "Voici le profil structurel du jeu de données :\n\n"
        + json.dumps(llm_profile, ensure_ascii=False, indent=2)
        + "\n\nProduis le plan d'analyse au format JSON demandé."
    )


# --------------------------------------------------------------------------
# Appel du modèle
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Appel du modèle — deux fournisseurs
# --------------------------------------------------------------------------
#
# L'architecture du pipeline ne dépend d'aucun fournisseur particulier : seul
# le format de la requête change. Garder les deux permet aussi de COMPARER
# leurs plans en fin de projet — un résultat de mémoire en soi.

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _detect_provider(model: str) -> str:
    m = model.lower()
    if m.startswith("gemini") or m.startswith("models/gemini"):
        return "google"
    return "anthropic"


def _build_request(system: str, user: str, model: str, api_key: str,
                   max_tokens: int, temperature: float):
    """Construit la requête HTTP selon le fournisseur détecté."""
    provider = _detect_provider(model)

    if provider == "google":
        # Google place la consigne système dans `system_instruction`, séparée
        # du tour utilisateur, et impose `responseMimeType` pour forcer le JSON.
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }
        url = GOOGLE_URL.format(model=model.replace("models/", ""))
        headers = {"content-type": "application/json", "x-goog-api-key": api_key}
    else:
        payload = {
            "model": model, "max_tokens": max_tokens, "temperature": temperature,
            "system": system, "messages": [{"role": "user", "content": user}],
        }
        url = ANTHROPIC_URL
        headers = {"content-type": "application/json", "x-api-key": api_key,
                   "anthropic-version": "2023-06-01"}

    return provider, urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers=headers, method="POST")


def _extract(provider: str, data: dict) -> tuple[str, dict]:
    """Extrait le texte et la consommation de tokens, quel que soit le format."""
    if provider == "google":
        candidats = data.get("candidates") or []
        texte = ""
        if candidats:
            texte = "".join(p.get("text", "")
                            for p in candidats[0].get("content", {}).get("parts", []))
        u = data.get("usageMetadata", {})
        usage = {"input_tokens": u.get("promptTokenCount"),
                 "output_tokens": u.get("candidatesTokenCount")}
        return texte, usage

    texte = "".join(b.get("text", "") for b in data.get("content", [])
                    if b.get("type") == "text")
    return texte, data.get("usage", {})


def call_llm(system: str, user: str, model: str, api_key: str,
             max_tokens: int = 8000, temperature: float = 0.2,
             log_dir: str | Path | None = None, tag: str = "planner") -> dict:
    """
    Appelle le modèle et journalise l'échange.

    Le fournisseur est déduit du nom du modèle : « gemini-… » pour Google AI
    Studio, tout le reste pour Anthropic.

    La journalisation n'est pas optionnelle : sans les prompts et réponses
    brutes archivés, impossible de démontrer que les mesures de fidélité
    factuelle portent sur des sorties réelles.
    """
    provider, req = _build_request(system, user, model, api_key,
                                   max_tokens, temperature)

    t0 = time.time()
    erreur = None
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read().decode("utf-8"))
        texte, usage = _extract(provider, data)
        if not texte:
            erreur = "réponse vide (quota atteint ou contenu filtré ?)"
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        erreur = f"HTTP {e.code} : {detail}"
        texte, usage = "", {}
    except Exception as e:
        erreur = f"{type(e).__name__} : {e}"
        texte, usage = "", {}

    latence = round(time.time() - t0, 2)

    if log_dir:
        d = Path(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        horodatage = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        (d / f"{horodatage}_{tag}.json").write_text(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provider": provider, "model": model, "tag": tag,
            "latency_s": latence, "temperature": temperature,
            "usage": usage, "error": erreur,
            "system_prompt": system, "user_prompt": user, "raw_response": texte,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"text": texte, "usage": usage, "latency_s": latence,
            "error": erreur, "provider": provider}


def parse_json_response(texte: str) -> tuple[dict | None, str | None]:
    """
    Extrait le JSON de la réponse.

    Malgré une consigne explicite, les modèles encadrent régulièrement leur
    sortie de ```json ou ajoutent une phrase d'introduction. On ne peut pas
    supposer que la consigne de format sera respectée à 100 %.
    """
    if not texte:
        return None, "réponse vide"

    t = texte.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    try:
        return json.loads(t), None
    except json.JSONDecodeError:
        pass

    # Repli : premier objet JSON équilibré trouvé dans le texte
    debut = t.find("{")
    if debut == -1:
        return None, "aucun JSON détecté"
    niveau, dans_texte, echap = 0, False, False
    for i, ch in enumerate(t[debut:], debut):
        if echap:
            echap = False
            continue
        if ch == "\\":
            echap = True
        elif ch == '"':
            dans_texte = not dans_texte
        elif not dans_texte:
            if ch == "{":
                niveau += 1
            elif ch == "}":
                niveau -= 1
                if niveau == 0:
                    try:
                        return json.loads(t[debut:i + 1]), None
                    except json.JSONDecodeError as e:
                        return None, f"JSON malformé : {e}"
    return None, "JSON incomplet"


# --------------------------------------------------------------------------
# Socle garanti
# --------------------------------------------------------------------------

def guaranteed_specs(mapping: dict) -> list[dict]:
    """
    Analyses injectées par le code, indépendamment du LLM.

    Un plan entièrement génératif peut, un jour donné, ne pas proposer
    l'évolution du chiffre d'affaires — les modèles varient d'un appel à
    l'autre. Devant un jury, c'est fatal. Le socle garantit un minimum ;
    le LLM complète avec ce qu'il découvre de spécifique.
    """
    m = {k: v for k, v in mapping.items() if v}
    out: list[dict] = []
    n = 0

    def ajouter(besoins, spec):
        nonlocal n
        if all(b in m for b in besoins):
            n += 1
            spec["id"] = f"base_{n:03d}"
            spec["source"] = "socle"
            out.append(spec)

    ajouter(["revenue", "order_date"], {
        "priority": 1, "title": "Évolution de votre chiffre d'affaires",
        "business_question": "Mon activité progresse-t-elle ou recule-t-elle ?",
        "measure": {"column": m.get("revenue"), "agg": "sum", "alias": "chiffre_d_affaires"},
        "dimensions": [{"column": m.get("order_date"), "type": "temporal", "grain": "month"}],
        "filters": [], "sort": {"by": "dimension", "order": "asc"}, "limit": None,
        "chart": "line",
        "business_rationale": "Indicateur maître : toute autre analyse s'interprète "
                              "relativement à cette tendance.",
        "pattern_hints": ["trend", "seasonality", "anomaly"],
    })

    ajouter(["revenue", "product_category"], {
        "priority": 2, "title": "Ce qui rapporte le plus",
        "business_question": "Quelles catégories font vraiment vivre ma boutique ?",
        "measure": {"column": m.get("revenue"), "agg": "sum", "alias": "chiffre_d_affaires"},
        "dimensions": [{"column": m.get("product_category"), "type": "categorical"}],
        "filters": [], "sort": {"by": "measure", "order": "desc"}, "limit": 15,
        "chart": "bar",
        "business_rationale": "Détecter une dépendance excessive à quelques catégories, "
                              "ou une dispersion qui disperse les efforts.",
        "pattern_hints": ["pareto_concentration", "long_tail"],
    })

    ajouter(["revenue", "order_date"], {
        "priority": 3, "title": "Combien dépensent vos clients",
        "business_question": "Mes clients dépensent-ils plus ou moins qu'avant ?",
        "measure": {"column": m.get("revenue"), "agg": "mean", "alias": "panier_moyen"},
        "dimensions": [{"column": m.get("order_date"), "type": "temporal", "grain": "month"}],
        "filters": [], "sort": {"by": "dimension", "order": "asc"}, "limit": None,
        "chart": "line",
        "business_rationale": "Un chiffre d'affaires stable avec un panier moyen en baisse "
                              "signale une érosion des prix compensée par le volume.",
        "pattern_hints": ["basket_erosion", "trend_decline", "divergence_with_volume"],
    })

    ajouter(["revenue", "order_date"], {
        "priority": 4, "title": "Vos meilleurs jours de la semaine",
        "business_question": "Quand mes clients achètent-ils ?",
        "measure": {"column": m.get("revenue"), "agg": "sum", "alias": "chiffre_d_affaires"},
        "dimensions": [{"column": m.get("order_date"), "type": "temporal", "grain": "dayofweek"}],
        "filters": [], "sort": {"by": "dimension", "order": "asc"}, "limit": None,
        "chart": "bar",
        "business_rationale": "Concentrer les campagnes et les stocks sur les jours forts.",
        "pattern_hints": ["day_of_week_peak"],
    })

    ajouter(["revenue", "product_id"], {
        "priority": 5, "title": "Vos produits les plus vendus",
        "business_question": "Quels produits marchent le mieux ?",
        "measure": {"column": m.get("revenue"), "agg": "sum", "alias": "chiffre_d_affaires"},
        "dimensions": [{"column": m.get("product_id"), "type": "categorical"}],
        "filters": [], "sort": {"by": "measure", "order": "desc"}, "limit": 15,
        "chart": "bar",
        "business_rationale": "Sécuriser l'approvisionnement des produits moteurs et "
                              "identifier ceux qui ne se vendent pas.",
        "pattern_hints": ["pareto_concentration", "long_tail_dead_stock"],
    })

    ajouter(["revenue", "region"], {
        "priority": 6, "title": "D'où viennent vos ventes",
        "business_question": "Quelles zones géographiques me rapportent le plus ?",
        "measure": {"column": m.get("revenue"), "agg": "sum", "alias": "chiffre_d_affaires"},
        "dimensions": [{"column": m.get("region"), "type": "categorical"}],
        "filters": [], "sort": {"by": "measure", "order": "desc"}, "limit": 12,
        "chart": "bar",
        "business_rationale": "Repérer les zones sous-exploitées où étendre la publicité, "
                              "ou les blocages logistiques ailleurs.",
        "pattern_hints": ["geographic_concentration"],
    })

    ajouter(["customer_id", "order_date", "revenue"], {
        "priority": 7, "title": "Vos clients reviennent-ils ?",
        "business_question": "Mes clients achètent-ils une seule fois ou plusieurs ?",
        "measure": {"column": m.get("customer_id"), "agg": "count_distinct",
                    "alias": "nombre_de_clients"},
        "dimensions": [{"column": m.get("order_date"), "type": "temporal", "grain": "month"}],
        "filters": [], "sort": {"by": "dimension", "order": "asc"}, "limit": None,
        "chart": "line",
        "business_rationale": "Un flux constant de nouveaux clients sans réachat signale "
                              "que l'effort d'acquisition ne se transforme pas en fidélité.",
        "pattern_hints": ["weak_retention", "single_purchase_dominance"],
    })

    return out


def make_plan(llm_profile: dict, mapping: dict, model: str, api_key: str,
              log_dir: str | Path | None = None,
              max_specs: int = 8) -> dict:
    """
    Plan complet = socle garanti (déterministe) ∪ specs découvertes (LLM).

    Si l'appel échoue, le socle seul est renvoyé : le système reste utilisable
    même sans le modèle. C'est la propriété qui rend une démonstration sûre.
    """
    socle = guaranteed_specs(mapping)

    reponse = call_llm(PLANNER_SYSTEM_PROMPT, build_user_prompt(llm_profile),
                       model=model, api_key=api_key, log_dir=log_dir, tag="planner")

    plan_llm, err_parse = (None, reponse["error"]) if reponse["error"] else \
        parse_json_response(reponse["text"])

    specs_llm = []
    comprehension = None
    if plan_llm:
        comprehension = plan_llm.get("dataset_understanding")
        for s in plan_llm.get("specs", []):
            s["source"] = "llm"
            specs_llm.append(s)

    return {
        "plan_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "provider": reponse.get("provider"),
        "dataset_understanding": comprehension,
        "specs": socle + specs_llm,
        "n_socle": len(socle),
        "n_llm": len(specs_llm),
        "max_specs_to_execute": max_specs,
        "llm_error": reponse["error"] or err_parse,
        "usage": reponse.get("usage", {}),
        "latency_s": reponse.get("latency_s"),
    }


def list_models(api_key: str, provider: str = "google") -> list[dict]:
    """
    Interroge l'API pour connaître les modèles réellement disponibles.

    Les noms d'API ne correspondent pas toujours aux libellés affichés dans
    l'interface (« Gemini 3.5 Flash Lite » peut s'appeler
    « gemini-3.5-flash-lite » ou « gemini-flash-lite-latest »), et l'offre
    évolue. Interroger l'API évite de deviner et supprime les erreurs 404.
    """
    if provider != "google":
        return [{"note": "Consultez docs.claude.com pour la liste Anthropic."}]

    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models",
        headers={"x-goog-api-key": api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return [{"error": f"{type(e).__name__} : {e}"}]

    out = []
    for m in data.get("models", []):
        if "generateContent" not in m.get("supportedGenerationMethods", []):
            continue
        out.append({
            "api_name": m["name"].replace("models/", ""),
            "display_name": m.get("displayName"),
            "input_limit": m.get("inputTokenLimit"),
            "output_limit": m.get("outputTokenLimit"),
        })
    return sorted(out, key=lambda x: x["api_name"])
