"""
Détection du secteur d'activité du commerçant.

Le fichier de ventes ne dit jamais explicitement ce que vend le commerce.
Pourtant, la même observation statistique appelle des interprétations
opposées selon la niche :

    « Vos ventes triplent en décembre »
      → vendeur d'articles cadeaux : normal, prévoir le stock 6 semaines avant
      → grossiste en fournitures BTP : anormal, chercher ce qui s'est passé

Le secteur est déduit des NOMS de produits et de catégories — jamais demandé
à l'utilisateur, mais toujours AFFICHÉ pour qu'il puisse corriger. Une niche
mal identifiée décale toutes les interprétations qui suivent, sans que le
commerçant puisse comprendre pourquoi les conseils tombent à côté.

Deux méthodes complémentaires :
  1. lexique déterministe — rapide, gratuit, transparent
  2. déduction par le LLM — plus fine sur les cas ambigus
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter

from .planner import call_llm, parse_json_response


# --------------------------------------------------------------------------
# Référentiel sectoriel
# --------------------------------------------------------------------------
#
# Chaque secteur porte ses mots-clés de reconnaissance, mais surtout son
# CONTEXTE MÉTIER : à quel moment de l'année les ventes montent, ce qui compte
# pour ce type de commerce, quels leviers ont du sens. C'est ce contexte qui
# nourrit l'interprétation.

SECTEURS = {
    "mode": {
        "label": "Mode et habillement",
        "mots": ["robe", "chemise", "pantalon", "veste", "jean", "pull", "tshirt",
                 "t-shirt", "chaussure", "basket", "sandale", "manteau", "jupe",
                 "vetement", "mode", "dress", "shirt", "shoes", "apparel",
                 "clothing", "fashion", "moda", "roupa", "calcado"],
        "saisonnalite": "Deux pics annuels aux changements de saison (mars-avril "
                        "et septembre-octobre), plus les soldes.",
        "enjeux": ["taux de retour élevé (tailles)", "rotation rapide des collections",
                   "gestion des fins de série"],
        "leviers": ["guide des tailles pour réduire les retours",
                    "déstockage avant nouvelle collection",
                    "ventes croisées haut/bas"],
    },
    "beaute": {
        "label": "Beauté et cosmétiques",
        "mots": ["parfum", "creme", "maquillage", "rouge a levres", "shampooing",
                 "soin", "serum", "beaute", "cosmetique", "vernis", "masque",
                 "beauty", "cosmetic", "perfume", "skincare", "makeup",
                 "beleza", "saude", "perfumaria"],
        "saisonnalite": "Pic marqué en novembre-décembre (cadeaux), "
                        "second pic à la fête des mères.",
        "enjeux": ["réachat régulier des consommables", "dates de péremption",
                   "fidélisation forte"],
        "leviers": ["abonnement ou relance au moment du réachat",
                    "coffrets cadeaux en fin d'année",
                    "échantillons pour découverte"],
    },
    "cadeaux_deco": {
        "label": "Cadeaux et décoration",
        "mots": ["cadeau", "decoration", "bougie", "vase", "cadre", "lanterne",
                 "ornement", "noel", "gift", "decor", "candle", "holder",
                 "christmas", "vintage", "retro", "heart", "hanging",
                 "presente", "decoracao"],
        "saisonnalite": "Très forte concentration sur novembre-décembre, "
                        "parfois plus de la moitié du chiffre d'affaires annuel.",
        "enjeux": ["dépendance extrême à la fin d'année",
                   "stock à constituer très en amont",
                   "creux de trésorerie de janvier à septembre"],
        "leviers": ["commander le stock de Noël dès septembre",
                    "développer les occasions hors Noël (mariages, anniversaires)",
                    "lisser la trésorerie sur l'année"],
    },
    "maison": {
        "label": "Maison et ameublement",
        "mots": ["meuble", "table", "chaise", "canape", "lit", "matelas", "linge",
                 "rideau", "cuisine", "ustensile", "casserole", "maison",
                 "furniture", "kitchen", "bedding", "home", "cama", "mesa",
                 "banho", "moveis"],
        "saisonnalite": "Pics au printemps et à la rentrée, "
                        "liés aux déménagements.",
        "enjeux": ["panier élevé mais achat peu fréquent",
                   "coût de livraison important", "cycle de décision long"],
        "leviers": ["facilités de paiement sur les gros paniers",
                    "ventes croisées d'accessoires",
                    "clarifier les frais de livraison en amont"],
    },
    "electronique": {
        "label": "Électronique et high-tech",
        "mots": ["telephone", "smartphone", "ordinateur", "tablette", "ecran",
                 "casque", "ecouteur", "chargeur", "cable", "batterie", "coque",
                 "informatique", "phone", "laptop", "computer", "headphone",
                 "charger", "electronics", "eletronicos", "informatica"],
        "saisonnalite": "Pics au Black Friday, à Noël et à la rentrée. "
                        "Très sensible aux sorties de nouveaux modèles.",
        "enjeux": ["obsolescence rapide des stocks", "marges faibles",
                   "concurrence sur le prix", "importance des accessoires"],
        "leviers": ["accessoires à forte marge en vente croisée",
                    "déstocker avant les nouvelles générations",
                    "garantie ou service comme différenciation"],
    },
    "sport": {
        "label": "Sport et loisirs",
        "mots": ["sport", "fitness", "velo", "course", "running", "yoga",
                 "musculation", "randonnee", "camping", "ballon", "raquette",
                 "bike", "gym", "outdoor", "esporte", "lazer"],
        "saisonnalite": "Pic en janvier (bonnes résolutions) et au printemps. "
                        "Creux estival sur le fitness intérieur.",
        "enjeux": ["forte saisonnalité par discipline",
                   "achat impulsif en janvier suivi d'abandon"],
        "leviers": ["offres de janvier préparées en décembre",
                    "programmes d'accompagnement pour fidéliser",
                    "équipements complémentaires"],
    },
    "alimentaire": {
        "label": "Alimentation et boissons",
        "mots": ["cafe", "the", "chocolat", "vin", "biere", "epicerie", "bio",
                 "snack", "boisson", "aliment", "food", "coffee", "tea", "wine",
                 "grocery", "organic", "alimentos", "bebidas"],
        "saisonnalite": "Ventes régulières, pics aux fêtes de fin d'année.",
        "enjeux": ["réachat fréquent", "dates limites de consommation",
                   "logistique du frais", "panier moyen faible"],
        "leviers": ["abonnement pour les consommables",
                    "seuil de livraison gratuite pour augmenter le panier",
                    "relance automatique au rythme de consommation"],
    },
    "bebe_enfant": {
        "label": "Bébé et enfant",
        "mots": ["bebe", "enfant", "jouet", "poussette", "couche", "biberon",
                 "puericulture", "peluche", "baby", "toy", "kids", "children",
                 "stroller", "brinquedos", "bebes"],
        "saisonnalite": "Pic à Noël sur les jouets, rentrée sur les fournitures.",
        "enjeux": ["clientèle qui sort du marché en grandissant",
                   "exigence forte sur la sécurité",
                   "achat souvent offert (grands-parents)"],
        "leviers": ["suivre l'âge de l'enfant pour proposer la tranche suivante",
                    "listes de naissance", "rassurer sur les normes"],
    },
    "bricolage_jardin": {
        "label": "Bricolage et jardin",
        "mots": ["outil", "perceuse", "visserie", "peinture", "jardin", "plante",
                 "graine", "tondeuse", "bricolage", "quincaillerie", "tool",
                 "garden", "diy", "hardware", "ferramentas", "jardim"],
        "saisonnalite": "Forte activité de mars à juin, creux hivernal.",
        "enjeux": ["saisonnalité très marquée", "produits volumineux",
                   "clientèle mixte particuliers et professionnels"],
        "leviers": ["préparer la saison dès février",
                    "gamme hivernale complémentaire",
                    "conseils d'usage pour rassurer les débutants"],
    },
    "sante_pharmacie": {
        "label": "Santé et parapharmacie",
        "mots": ["pharmacie", "medicament", "complement", "vitamine", "hygiene",
                 "medical", "orthopedie", "health", "supplement", "pharmacy",
                 "saude", "farmacia"],
        "saisonnalite": "Pic hivernal, second pic au printemps (allergies).",
        "enjeux": ["réachat très régulier", "réglementation",
                   "fidélité forte quand la confiance est établie"],
        "leviers": ["relance au rythme du renouvellement",
                    "conseils personnalisés", "programme de fidélité"],
    },
    "papeterie_livres": {
        "label": "Papeterie, livres et culture",
        "mots": ["livre", "cahier", "stylo", "papeterie", "agenda", "carte",
                 "bureau", "book", "stationery", "pen", "notebook", "office",
                 "livros", "papelaria"],
        "saisonnalite": "Pic à la rentrée de septembre et à Noël.",
        "enjeux": ["panier faible", "forte concurrence",
                   "produits peu différenciés"],
        "leviers": ["lots et packs pour augmenter le panier",
                    "personnalisation comme différenciation",
                    "anticiper la rentrée dès juillet"],
    },
    "bijoux_accessoires": {
        "label": "Bijoux et accessoires",
        # « sac », « bag » et « ring » sont volontairement absents : ils
        # attrapent les sacs de courses et les anneaux de décoration. Un
        # mot-clé trop général contamine le score des autres secteurs.
        "mots": ["bijou", "collier", "bracelet", "bague", "montre", "boucle",
                 "ceinture", "portefeuille", "maroquinerie", "jewel", "jewellery",
                 "necklace", "bracelet", "earring", "handbag", "sac a main",
                 "relogios", "bolsas"],
        "saisonnalite": "Pics à Noël, Saint-Valentin et fête des mères.",
        "enjeux": ["achat souvent offert", "importance de la présentation",
                   "marge élevée"],
        "leviers": ["emballage cadeau soigné",
                    "campagnes calées sur les fêtes",
                    "gravure ou personnalisation"],
    },
}

SECTEUR_INCONNU = {
    "label": "Commerce généraliste",
    "saisonnalite": "Aucune saisonnalité type identifiée.",
    "enjeux": ["catalogue varié"],
    "leviers": ["analyser les catégories séparément"],
}


def _sans_accents(t: str) -> str:
    n = unicodedata.normalize("NFKD", str(t).lower())
    return "".join(c for c in n if not unicodedata.combining(c))


# --------------------------------------------------------------------------
# Méthode 1 — lexique
# --------------------------------------------------------------------------

def detecter_par_lexique(echantillon: list[str]) -> dict:
    """
    Compte les correspondances de mots-clés dans les noms de produits.

    Rapide, gratuit et explicable : on peut montrer à l'utilisateur quels mots
    ont conduit à la conclusion. Limite : inopérant sur des libellés très
    spécifiques ou dans une langue non couverte.
    """
    textes = [_sans_accents(t) for t in echantillon if t]
    if not textes:
        return {"secteur": None, "confiance": 0.0, "methode": "lexique"}

    corpus = " ".join(textes)
    scores = Counter()
    preuves: dict[str, list[str]] = {}

    for cle, s in SECTEURS.items():
        for mot in s["mots"]:
            m = _sans_accents(mot)
            n = len(re.findall(rf"\b{re.escape(m)}", corpus))
            if n:
                scores[cle] += n
                preuves.setdefault(cle, []).append(mot)

    if not scores:
        return {"secteur": None, "confiance": 0.0, "methode": "lexique"}

    cle, n = scores.most_common(1)[0]
    second = scores.most_common(2)[1][1] if len(scores) > 1 else 0

    # La confiance mesure la DOMINANCE du premier secteur sur le second, non
    # sa part du total. Diviser par le total pénalisait injustement les
    # catalogues riches : un fichier où beaucoup de mots-clés répondent voyait
    # son secteur dominant dilué, alors même qu'il était clairement identifié.
    if second == 0:
        confiance = 0.95 if n >= 3 else 0.6
    else:
        ecart = n / (n + second)          # 0,5 = égalité, 1,0 = domination
        confiance = max(0.0, (ecart - 0.5) * 2.4)
        if n < 3:
            confiance *= 0.6

    return {
        "secteur": cle,
        "label": SECTEURS[cle]["label"],
        "confiance": round(min(0.95, confiance), 2),
        "methode": "lexique",
        "mots_trouves": preuves.get(cle, [])[:8],
        "alternatives": [{"secteur": k, "label": SECTEURS[k]["label"], "score": v}
                         for k, v in scores.most_common(4)[1:]],
    }


# --------------------------------------------------------------------------
# Méthode 2 — déduction par le modèle
# --------------------------------------------------------------------------

SECTEUR_PROMPT = """\
Tu identifies le secteur d'activité d'un commerce en ligne à partir des noms
de ses produits ou catégories.

Réponds UNIQUEMENT par un JSON de cette forme, sans texte autour :

{
  "secteur": "<un identifiant parmi la liste fournie, ou 'autre'>",
  "label": "<nom du secteur en français, 2 à 4 mots>",
  "confiance": <nombre entre 0 et 1>,
  "description": "<une phrase : que vend ce commerce, et à qui>",
  "saisonnalite": "<une phrase : à quels moments de l'année ses ventes montent, et pourquoi>",
  "enjeux": ["<enjeu métier 1>", "<enjeu 2>", "<enjeu 3>"]
}

Si les noms de produits ne permettent pas de conclure, mets "autre" avec une
confiance basse. N'invente pas un secteur pour combler le doute."""


def detecter_par_llm(echantillon: list[str], model: str, api_key: str,
                     log_dir=None) -> dict:
    """Déduction par le modèle. Plus fine que le lexique sur les cas ambigus."""
    if not echantillon:
        return {"secteur": None, "confiance": 0.0, "methode": "llm"}

    user = (
        "Identifiants de secteurs disponibles :\n"
        + ", ".join(SECTEURS) + ", autre\n\n"
        "Noms de produits ou catégories de ce commerce :\n"
        + "\n".join(f"- {t}" for t in echantillon[:60])
        + "\n\nQuel est son secteur d'activité ?"
    )

    rep = call_llm(SECTEUR_PROMPT, user, model=model, api_key=api_key,
                   max_tokens=800, temperature=0.1,
                   log_dir=log_dir, tag="secteur")
    if rep["error"]:
        return {"secteur": None, "confiance": 0.0, "methode": "llm",
                "error": rep["error"]}

    parsed, err = parse_json_response(rep["text"])
    if not parsed:
        return {"secteur": None, "confiance": 0.0, "methode": "llm",
                "error": err}

    parsed["methode"] = "llm"
    parsed.setdefault("confiance", 0.5)
    return parsed


# --------------------------------------------------------------------------
# Point d'entrée
# --------------------------------------------------------------------------

def echantillon_produits(df, mapping: dict, taille: int = 60) -> list[str]:
    """
    Noms de produits et catégories les plus fréquents.

    On prend les plus fréquents plutôt qu'un tirage aléatoire : ils
    caractérisent mieux l'activité principale du commerce.

    Un repli est indispensable : beaucoup de fichiers n'ont ni « catégorie »
    ni « produit » explicitement nommés — Online Retail II a « Description »
    et « StockCode ». Sans repli, la détection échoue précisément sur les
    fichiers où l'information est présente mais mal étiquetée.
    """
    def parlant(v: str) -> bool:
        """Un libellé porte du sens ; un code produit n'en porte aucun.

        « 85123A » ou « REF-001 » ne disent rien du secteur. Les retenir
        empêche la détection sur des fichiers qui contiennent pourtant une
        colonne de description exploitable juste à côté."""
        v = v.strip()
        if len(v) <= 2 or v.replace(".", "").isdigit():
            return False
        lettres = sum(c.isalpha() for c in v)
        chiffres = sum(c.isdigit() for c in v)
        if chiffres and lettres <= chiffres:
            return False
        # Un code : court, sans espace, mêlant lettres et chiffres
        if " " not in v and len(v) <= 10 and chiffres:
            return False
        return lettres >= 3

    noms = []
    for cle in ("product_category", "product_id"):
        col = mapping.get(cle)
        if col and col in df.columns:
            vc = df[col].dropna().astype(str).value_counts()
            noms += [v for v in vc.head(taille).index if parlant(v)]

    if len(noms) < 5:
        # Repli : toute colonne textuelle non mappée à un autre rôle
        deja = set(mapping.values())
        for col in df.columns:
            if col in deja or col in noms:
                continue
            s = df[col].dropna()
            if s.empty or len(s) < 10:
                continue
            s = s.astype(str)
            # Des libellés : majoritairement du texte, pas des codes courts
            if s.str.len().median() < 5:
                continue
            if s.str.replace(r"[\s\-_.]", "", regex=True).str.isalpha().mean() < 0.5:
                continue
            vc = s.value_counts()
            if 3 <= len(vc) <= len(s) * 0.5:
                noms += [v for v in vc.head(taille).index if parlant(v)]
                if len(noms) >= 20:
                    break

    return noms[:taille]


def detecter_secteur(df, mapping: dict, model: str | None = None,
                     api_key: str | None = None, log_dir=None) -> dict:
    """
    Identifie le secteur. Le lexique d'abord, le modèle si le doute persiste.

    Le résultat est toujours accompagné de sa méthode, de sa confiance et des
    indices retenus : l'utilisateur doit pouvoir juger et corriger.
    """
    echantillon = echantillon_produits(df, mapping)
    if not echantillon:
        return {**SECTEUR_INCONNU, "secteur": None, "confiance": 0.0,
                "methode": "aucune donnée produit", "indices": [],
                "alternatives": [], "a_confirmer": True,
                "echantillon_utilise": [], "description": None,
                "message": "Aucun nom de produit exploitable dans votre fichier."}

    res = detecter_par_lexique(echantillon)

    # Le modèle n'est appelé que si le lexique hésite : inutile de payer un
    # appel quand les mots-clés sont sans ambiguïté.
    if res["confiance"] < 0.6 and model and api_key:
        llm = detecter_par_llm(echantillon, model, api_key, log_dir)
        if llm.get("confiance", 0) > res["confiance"]:
            res = llm

    cle = res.get("secteur")

    # Sous 0,35 de confiance, plusieurs secteurs sont à égalité : c'est un
    # catalogue mixte, pas une niche. Trancher au hasard donnerait un contexte
    # métier faux, donc des interprétations décalées — pire que pas de contexte.
    if res.get("confiance", 0) < 0.35:
        alt = res.get("alternatives") or []
        return {**SECTEUR_INCONNU, "secteur": None,
                "label": "Commerce multi-catégories",
                "confiance": round(res.get("confiance", 0), 2),
                "methode": res.get("methode"), "indices": res.get("mots_trouves", []),
                "alternatives": alt, "a_confirmer": True,
                "description": "Catalogue réparti sur plusieurs univers produits.",
                "saisonnalite": "Analyser chaque catégorie séparément : "
                                "les rythmes diffèrent d'un univers à l'autre.",
                "enjeux": ["catalogue hétérogène",
                           "saisonnalités qui se compensent"],
                "leviers": ["comparer les catégories entre elles",
                            "identifier celle qui porte réellement l'activité"],
                "echantillon_utilise": echantillon[:10]}

    contexte = SECTEURS.get(cle, SECTEUR_INCONNU)

    return {
        "secteur": cle,
        "label": res.get("label") or contexte["label"],
        "confiance": res.get("confiance", 0.0),
        "methode": res.get("methode"),
        "description": res.get("description"),
        "saisonnalite": res.get("saisonnalite") or contexte["saisonnalite"],
        "enjeux": res.get("enjeux") or contexte["enjeux"],
        "leviers": contexte.get("leviers", []),
        "indices": res.get("mots_trouves", []),
        "alternatives": res.get("alternatives", []),
        "a_confirmer": res.get("confiance", 0) < 0.7,
        "echantillon_utilise": echantillon[:10],
    }


def contexte_pour_interprete(secteur: dict) -> str:
    """
    Bloc de contexte inséré dans le prompt de l'interprète.

    C'est ce qui transforme « vos ventes triplent en décembre » en « c'est le
    rythme normal de votre secteur » ou en « c'est inhabituel, cherchez la
    cause » selon la niche.
    """
    if not secteur.get("secteur"):
        return ""

    lignes = [
        "CONTEXTE DU COMMERCE",
        f"Secteur : {secteur['label']}",
    ]
    if secteur.get("description"):
        lignes.append(f"Activité : {secteur['description']}")
    if secteur.get("saisonnalite"):
        lignes.append(f"Rythme habituel du secteur : {secteur['saisonnalite']}")
    if secteur.get("enjeux"):
        lignes.append("Enjeux propres à ce métier : " + " ; ".join(secteur["enjeux"]))
    if secteur.get("leviers"):
        lignes.append("Leviers qui fonctionnent dans ce secteur : "
                      + " ; ".join(secteur["leviers"]))
    lignes.append(
        "Utilise ce contexte pour situer les chiffres : un pic attendu dans ce "
        "secteur n'est pas une anomalie, et une action pertinente ailleurs peut "
        "ne pas l'être ici. N'invente aucun chiffre à partir de ce contexte."
    )
    return "\n".join(lignes)
