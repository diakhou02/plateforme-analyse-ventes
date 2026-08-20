"""
Interprétation : le LLM verbalise des faits déjà calculés.

Étape [7] du pipeline. Le modèle reçoit `facts.json` — jamais les données —
et produit trois blocs de texte destinés au commerçant :

    CONSTAT  →  DIAGNOSTIC  →  ACTIONS

Le graphique est affiché AVANT ces textes : il sert de preuve. Le commerçant
voit la courbe descendre, puis lit pourquoi et quoi faire.

CE QUE LE MODÈLE NE FAIT PAS
- il ne calcule rien : tous les chiffres viennent de facts.json ;
- il n'invente aucune recommandation : il reformule les actions du pattern
  détecté par le code, en les adaptant aux chiffres réels.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .planner import call_llm, parse_json_response

# --------------------------------------------------------------------------
# Prompt système
# --------------------------------------------------------------------------

INTERPRETER_SYSTEM_PROMPT = """\
Tu es un conseiller commercial qui s'adresse à un commerçant en ligne. Cette
personne n'est pas analyste : elle ne connaît ni le vocabulaire statistique,
ni les indicateurs techniques. Elle veut savoir ce qui marche, ce qui ne
marche pas, et quoi faire.

On te fournit les RÉSULTATS DÉJÀ CALCULÉS d'une analyse (JSON), accompagnés
d'un graphique que l'utilisateur a sous les yeux au moment de te lire.

INTERDICTIONS ABSOLUES
- N'invente aucun chiffre. Tu ne peux citer QUE des valeurs présentes
  littéralement dans le JSON fourni. Aucun calcul de ta part, même simple :
  ni pourcentage, ni somme, ni moyenne, ni différence.
- N'utilise aucun jargon : ni « RFM », ni « cohorte », ni « Pareto », ni
  « corrélation », ni « écart-type », ni « médiane », ni « régression », ni
  « tendance linéaire », ni « r² », ni « anomalie ».
- N'invente aucune cause que les données ne permettent pas d'établir.
  Formule les hypothèses au conditionnel.
- Ne recommande que des actions issues du champ "actions" du pattern détecté.
  Tu peux les reformuler et les adapter aux chiffres réels, mais tu ne dois
  pas en inventer d'autres.

STRUCTURE DE RÉPONSE — exactement ces trois blocs, dans cet ordre

1. CONSTAT — 2 à 3 phrases. Le fait, avec les chiffres clés. Le commerçant
   doit retrouver dans ta phrase ce qu'il voit sur le graphique.

2. DIAGNOSTIC — 2 à 4 phrases. Pourquoi c'est important pour son activité, et
   quelles sont les causes probables. Au conditionnel si non prouvé.
   Si les données ne révèlent aucun problème, dis-le et explique ce qui va bien.

3. ACTIONS — 2 ou 3 actions concrètes, formulées à l'impératif. Chacune doit
   être réalisable par un commerçant seul, sans outil supplémentaire, dans les
   30 jours. Pas de « améliorer votre stratégie » : dis quoi faire précisément.

SI le JSON contient "last_period_incomplete": true, précise dans le constat que
la dernière période affichée est partielle et ne doit pas être comparée aux
autres.

RÈGLE IMPÉRATIVE SUR LES TENDANCES — lis "trend_direction" avant d'écrire :

- "stable" : tu n'as PAS le droit d'annoncer une hausse ni une baisse, même si
  "change_pct" est important. Ce chiffre compare deux périodes isolées d'une
  série qui n'a aucune direction nette. Dis que l'activité est STABLE, et
  utilise "volatilite_pct" pour indiquer l'amplitude normale des variations
  d'une période à l'autre. N'utilise PAS "change_pct" dans ta phrase.
- "baisse" ou "hausse" : la tendance est établie, tu peux l'annoncer et citer
  "change_pct".

De même, ne désigne un « meilleur » jour, produit ou catégorie que si son écart
au reste est net. Quand les valeurs sont proches, dis qu'elles sont équilibrées
plutôt que de sacrer un gagnant : nommer un vainqueur là où il n'y en a pas
conduit le commerçant à des décisions fondées sur du hasard.

SI un CONTEXTE DU COMMERCE t'est fourni, utilise-le pour SITUER les chiffres :
un pic attendu dans ce secteur se commente autrement qu'un pic inexpliqué, et
une action pertinente ailleurs peut ne pas l'être ici. Ce contexte ne contient
AUCUN chiffre : ne t'en sers jamais pour en produire.

TON
Direct, concret, bienveillant. Vouvoiement. Phrases courtes. Pas de préambule,
pas de formule de politesse, pas de conclusion.

FORMAT DE SORTIE
Réponds UNIQUEMENT par un objet JSON valide, sans texte avant ni après,
sans bloc de code Markdown :

{
  "constat": "...",
  "diagnostic": "...",
  "actions": ["...", "...", "..."]
}"""


SYNTHESIS_SYSTEM_PROMPT = """\
Tu es un conseiller commercial. On te fournit le résumé de plusieurs analyses
déjà réalisées sur la boutique en ligne d'un commerçant, avec les problèmes
détectés dans chacune.

Ta mission : produire une SYNTHÈSE GÉNÉRALE qui relie ces observations entre
elles et hiérarchise ce qui est urgent.

INTERDICTIONS ABSOLUES
- N'invente aucun chiffre : n'utilise que ceux présents dans le JSON fourni.
- Aucun jargon technique.
- Ne recommande que des actions issues des analyses fournies.
- Si une analyse porte "tendance": "stable", tu ne dois annoncer NI hausse NI
  baisse pour elle. Dis que l'activité se maintient, et sers-toi de
  "variation_normale_pct" pour donner l'amplitude habituelle des variations.
- Si un PÉRIMÈTRE t'est fourni et que des lignes ont été écartées, mentionne-le
  en une phrase dans "situation" : le commerçant doit savoir que les chiffres
  ne portent pas sur la totalité de son fichier.
- Ne désigne un « meilleur » jour, produit ou catégorie que si l'écart au reste
  est net. Quand les valeurs sont proches, dis qu'elles sont équilibrées : un
  vainqueur désigné au hasard conduit à des décisions fondées sur du bruit.

STRUCTURE
1. "situation" — 3 à 4 phrases. Où en est ce commerce aujourd'hui ? Relie les
   observations : si le chiffre d'affaires baisse ET que les clients ne
   reviennent pas, dis que l'un explique probablement l'autre.
2. "priorites" — exactement 3 priorités classées, de la plus urgente à la
   moins urgente. Chacune : un titre court et une phrase d'explication.
3. "premiere_action" — LA seule chose à faire cette semaine. Une phrase.

TON
Direct, sans complaisance mais sans alarmisme. Vouvoiement.

FORMAT DE SORTIE — JSON uniquement, sans texte autour :
{
  "situation": "...",
  "priorites": [
    {"titre": "...", "explication": "..."},
    {"titre": "...", "explication": "..."},
    {"titre": "...", "explication": "..."}
  ],
  "premiere_action": "..."
}"""


# --------------------------------------------------------------------------
# Préparation du contexte transmis au modèle
# --------------------------------------------------------------------------

def facts_for_llm(facts: dict, max_points: int = 30) -> dict:
    """
    Allège les faits avant envoi.

    Une série de 400 jours n'apporte rien de plus au modèle que ses statistiques
    et une trentaine de points. On échantillonne en conservant impérativement
    le premier et le dernier point, qui portent l'évolution.
    """
    pts = facts["data_points"]
    if len(pts) > max_points:
        pas = len(pts) // (max_points - 2)
        pts = [pts[0]] + pts[1:-1:pas][:max_points - 2] + [pts[-1]]

    return {
        "titre": facts["title"],
        "question": facts.get("business_question"),
        "mesure": facts.get("measure_label"),
        "unite": facts.get("unit"),
        "axe": facts.get("dimension_labels", [None])[0],
        "points": pts,
        "statistiques": facts["computed_stats"],
        "lecture_tendance": facts["computed_stats"].get("trend_comment"),
        "problemes_detectes": [
            {"nom": p["pattern_id"], "preuve": p["evidence"],
             "causes_possibles": p["causes"], "actions": p["actions"]}
            for p in facts.get("detected_patterns", [])
        ],
        "couverture": facts.get("coverage", {}),
    }


def interpret_one(facts: dict, model: str, api_key: str,
                  log_dir=None, temperature: float = 0.3,
                  secteur: dict | None = None) -> dict:
    """
    Interprète une analyse. Retourne constat / diagnostic / actions.

    Le `secteur` change la lecture des mêmes chiffres : un triplement des
    ventes en décembre est le rythme normal d'un vendeur d'articles cadeaux,
    et une anomalie à élucider chez un grossiste en fournitures.
    """
    contexte = facts_for_llm(facts)

    bloc_secteur = ""
    if secteur:
        from .sector import contexte_pour_interprete
        c = contexte_pour_interprete(secteur)
        if c:
            bloc_secteur = c + "\n\n"

    user = (
        bloc_secteur
        + "Voici les résultats calculés de l'analyse :\n\n"
        + json.dumps(contexte, ensure_ascii=False, indent=2)
        + "\n\nRédige le constat, le diagnostic et les actions."
    )

    rep = call_llm(INTERPRETER_SYSTEM_PROMPT, user, model=model, api_key=api_key,
                   max_tokens=2000, temperature=temperature,
                   log_dir=log_dir, tag=f"interprete_{facts.get('spec_id')}")

    if rep["error"]:
        return {"error": rep["error"], "latency_s": rep["latency_s"]}

    parsed, err = parse_json_response(rep["text"])
    if parsed is None:
        return {"error": f"format invalide : {err}", "raw": rep["text"][:400]}

    return {
        "spec_id": facts.get("spec_id"),
        "constat": parsed.get("constat", ""),
        "diagnostic": parsed.get("diagnostic", ""),
        "actions": parsed.get("actions", []),
        "latency_s": rep["latency_s"],
        "usage": rep.get("usage", {}),
    }


def interpret_all(facts_list: list, model: str, api_key: str,
                  log_dir=None, secteur: dict | None = None) -> list:
    return [interpret_one(f, model, api_key, log_dir, secteur=secteur)
            for f in facts_list]


# --------------------------------------------------------------------------
# Synthèse transversale
# --------------------------------------------------------------------------

def build_synthesis(facts_list: list, model: str, api_key: str,
                    log_dir=None, secteur: dict | None = None,
                    perimetre: dict | None = None) -> dict:
    """
    Synthèse générale. Seul endroit où le modèle relie les analyses entre elles.

    Il ne reçoit que les titres, les statistiques essentielles et les problèmes
    détectés — pas les séries complètes.
    """
    resume = []
    for f in facts_list:
        st = f["computed_stats"]
        entree = {
            "analyse": f["title"],
            "problemes": [p["pattern_id"] for p in f.get("detected_patterns", [])],
        }
        # La QUALIFICATION prime sur la variation brute. Sans elle, la
        # synthèse annonçait « baisse de 14,9 % » sur une série dont la pente
        # est positive et le r² de 0,005 : le chiffre est exact, sa lecture
        # est fausse. C'est le même défaut que sur les analyses individuelles,
        # corrigé là mais oublié ici.
        direction = st.get("trend_direction")
        if direction:
            entree["tendance"] = direction
            if direction == "stable":
                entree["remarque"] = ("série stable, ne pas annoncer de hausse "
                                      "ni de baisse")
                entree["variation_normale_pct"] = st.get("volatilite_pct")
            else:
                entree["change_pct"] = st.get("change_pct")
        for cle in ("top20_share_pct", "top1_share_pct", "total"):
            if cle in st:
                entree[cle] = st[cle]
        if not direction and "change_pct" in st:
            entree["change_pct"] = st["change_pct"]
        if st.get("max"):
            entree["plus_fort"] = st["max"]
        resume.append(entree)

    bloc_secteur = ""
    if secteur:
        from .sector import contexte_pour_interprete
        c = contexte_pour_interprete(secteur)
        if c:
            bloc_secteur = c + "\n\n"

    bloc_perimetre = ""
    if perimetre:
        # Le modèle doit savoir sur quelles données il commente : parler du
        # « chiffre d'affaires » sans préciser qu'un dixième des lignes a été
        # écarté rend le rapport invérifiable.
        bloc_perimetre = ("PÉRIMÈTRE DE L'ANALYSE\n"
                          + json.dumps(perimetre, ensure_ascii=False) + "\n\n")

    user = (bloc_secteur + bloc_perimetre
            + "Voici le résumé des analyses réalisées :\n\n"
            + json.dumps(resume, ensure_ascii=False, indent=2)
            + "\n\nProduis la synthèse générale.")

    rep = call_llm(SYNTHESIS_SYSTEM_PROMPT, user, model=model, api_key=api_key,
                   max_tokens=2000, temperature=0.3,
                   log_dir=log_dir, tag="synthese")

    if rep["error"]:
        return {"error": rep["error"]}

    parsed, err = parse_json_response(rep["text"])
    if parsed is None:
        return {"error": f"format invalide : {err}"}

    parsed["latency_s"] = rep["latency_s"]
    parsed["usage"] = rep.get("usage", {})
    parsed["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return parsed
