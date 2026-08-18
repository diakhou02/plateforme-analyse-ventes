"""
Mapping sémantique : associer chaque colonne à son rôle métier e-commerce.

Étape [2b] du pipeline, entre le profilage et le diagnostic.

C'est LE POINT DE RUPTURE de l'expérience utilisateur : si l'import échoue,
le commerçant abandonne avant d'avoir vu la moindre analyse. Un export Shopify,
WooCommerce ou un fichier Excel bricolé n'a jamais le même schéma.

Trois signaux combinés, aucun décisif seul :
  1. le NOM de la colonne (dictionnaire de synonymes multilingue)
  2. le RÔLE structurel issu du profilage (measure, temporal, identifier…)
  3. le CONTENU (plage de valeurs, motifs, cardinalité)

Le résultat est toujours accompagné d'un score de confiance et soumis à
l'utilisateur pour validation — jamais appliqué silencieusement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .profiler import _tokens, normalize_name

# --------------------------------------------------------------------------
# Rôles métier reconnus
# --------------------------------------------------------------------------
# `required` : sans ce rôle, presque aucune analyse n'est possible
# `structural_roles` : rôles du profileur compatibles — filtre les absurdités
#                      (une date ne peut pas être un montant)

CONCEPTS = {
    "order_id": {
        "label": "Numéro de commande",
        "required": False,
        "structural_roles": {"identifier", "categorical", "high_cardinality_cat", "measure"},
        "synonyms": [
            "order id", "order number", "order", "commande", "num commande",
            "numero commande", "no commande", "id commande", "ref commande",
            "reference commande", "order ref", "transaction id", "invoice",
            "facture", "num facture", "bon commande", "order_id", "orderid",
            "pedido", "bestellung", "ordine",
        ],
    },
    "customer_id": {
        "label": "Identifiant client",
        "required": False,
        "structural_roles": {"identifier", "categorical", "high_cardinality_cat"},
        "synonyms": [
            "customer id", "client id", "id client", "customer", "client",
            "acheteur", "buyer", "user id", "utilisateur", "compte", "account",
            "email", "e mail", "mail", "customer email", "customer_unique_id",
            "cliente", "kunde", "num client", "code client",
        ],
    },
    "order_date": {
        "label": "Date de commande",
        "required": True,
        "structural_roles": {"temporal"},
        "synonyms": [
            "order date", "date commande", "date achat", "purchase date",
            "date", "created at", "date creation", "timestamp", "jour",
            "transaction date", "date vente", "sale date", "paid at",
            "order purchase timestamp", "fecha", "datum", "data",
        ],
        # Une date de livraison n'est pas une date de commande.
        "penalize": ["livraison", "delivery", "shipping", "expedition", "shipped",
                     "estimated", "prevue", "retour", "return", "approved",
                     "carrier", "limite"],
    },
    "revenue": {
        "label": "Montant de la vente",
        "required": True,
        "structural_roles": {"measure"},
        "synonyms": [
            "price", "prix", "montant", "amount", "total", "revenue",
            "chiffre affaires", "ca", "total ttc", "total ht", "valeur",
            "value", "sales", "vente", "prix total", "montant total",
            "order total", "line total", "subtotal", "sous total", "payment value",
            "precio", "importe", "betrag", "preis", "prezzo", "net",
        ],
        # Frais de port et remises ne sont pas le CA.
        "penalize": ["unitaire", "unit price", "cout", "cost", "achat", "purchase price",
                     "frais", "freight", "shipping", "port", "livraison", "tax",
                     "taxe", "tva", "remise", "discount", "reduction"],
    },
    "quantity": {
        "label": "Quantité",
        "required": False,
        "structural_roles": {"measure", "categorical"},
        "synonyms": [
            "quantity", "quantite", "qte", "qty", "nombre", "nb", "count",
            "units", "unites", "volume", "pieces", "articles", "items",
            "cantidad", "menge", "quantita",
        ],
        # « order_item_id » (Olist) est un numéro de séquence d'article, pas
        # une quantité. Le confondre faisait requalifier le montant en prix
        # unitaire et vidait le chiffre d'affaires.
        "penalize": ["id", "identifiant", "numero", "sequence", "rang", "index"],
    },
    "product_id": {
        "label": "Produit",
        "required": False,
        "structural_roles": {"identifier", "categorical", "high_cardinality_cat"},
        "synonyms": [
            "product id", "produit", "product", "article", "item", "sku",
            "reference produit", "ref produit", "code produit", "designation",
            "libelle", "nom produit", "product name", "titre", "title",
            "description", "libelle produit", "intitule", "nom article",
            "stockcode", "stock code", "descripcion",
            "producto", "artikel", "prodotto", "ean", "isbn", "gtin",
        ],
    },
    "product_category": {
        "label": "Catégorie de produit",
        "required": False,
        "structural_roles": {"categorical", "high_cardinality_cat"},
        "synonyms": [
            "category", "categorie", "cat", "famille", "family", "type",
            "rayon", "segment", "gamme", "collection", "univers", "groupe",
            "product category name", "product type", "categoria", "kategorie",
        ],
    },
    "order_status": {
        "label": "Statut de la commande",
        "required": False,
        "structural_roles": {"categorical", "boolean_flag", "constant"},
        "synonyms": [
            "status", "statut", "etat", "state", "order status", "situation",
            "fulfillment status", "financial status", "payment status",
            "estado", "zustand", "stato",
        ],
        # Le contenu prime : voir `_status_evidence`
    },
    "unit_price": {
        "label": "Prix unitaire",
        "required": False,
        "structural_roles": {"measure"},
        "synonyms": [
            "prix unitaire", "unit price", "prix un", "pu", "prix article",
            "price per unit", "tarif unitaire", "prix ht unitaire",
            "unitario", "einzelpreis",
        ],
        "penalize": ["total", "montant total", "somme", "frais"],
    },
    "delivery_date": {
        "label": "Date de livraison",
        "required": False,
        "structural_roles": {"temporal"},
        "synonyms": [
            "date livraison", "delivery date", "date expedition", "shipped at",
            "date reception", "delivered at", "date envoi", "ship date",
            "order delivered customer date", "fecha entrega",
        ],
    },
    "order_total": {
        "label": "Total de la commande",
        "required": False,
        "structural_roles": {"measure"},
        "synonyms": [
            "total commande", "order total", "montant commande", "total ttc",
            "grand total", "total general", "montant paye", "paid amount",
        ],
        "penalize": ["ligne", "line", "article", "unitaire", "frais", "port"],
    },
    "discount": {
        "label": "Remise",
        "required": False,
        "structural_roles": {"measure", "categorical"},
        "synonyms": [
            "remise", "discount", "reduction", "rabais", "promo", "promotion",
            "taux remise", "discount rate", "discount pct", "descuento",
            "rabatt", "desconto",
        ],
    },
    "shipping_cost": {
        "label": "Frais de livraison",
        "required": False,
        "structural_roles": {"measure"},
        "synonyms": [
            "frais de port", "frais livraison", "shipping", "freight",
            "freight value", "port", "livraison cout", "shipping cost",
            "frais expedition", "gastos envio",
        ],
    },
    "region": {
        "label": "Zone géographique",
        "required": False,
        "structural_roles": {"categorical", "high_cardinality_cat"},
        "synonyms": [
            "region", "ville", "city", "pays", "country", "departement",
            "state", "province", "zone", "secteur", "localite", "commune",
            "customer state", "customer city", "shipping city", "adresse",
            "address", "code postal", "zip", "postal", "ciudad", "stadt",
        ],
    },
    "payment_method": {
        "label": "Moyen de paiement",
        "required": False,
        "structural_roles": {"categorical", "boolean_flag"},
        "synonyms": [
            "payment", "paiement", "mode paiement", "moyen paiement",
            "payment type", "payment method", "carte", "card", "reglement",
            "pago", "zahlung",
        ],
    },
    "channel": {
        "label": "Canal de vente",
        "required": False,
        "structural_roles": {"categorical", "boolean_flag"},
        "synonyms": [
            "channel", "canal", "source", "origine", "origin", "referrer",
            "boutique", "store", "shop", "marketplace", "plateforme",
            "utm source", "medium",
        ],
    },
}

REQUIRED = [k for k, v in CONCEPTS.items() if v["required"]]

# Valeurs typiques d'une colonne de statut — preuve par le contenu
STATUS_VALUES = {
    "livre", "livree", "delivered", "shipped", "expedie", "expediee",
    "annule", "annulee", "cancelled", "canceled", "pending", "en attente",
    "paid", "paye", "payee", "refunded", "rembourse", "processing",
    "en cours", "complete", "completed", "termine", "invoiced", "unavailable",
    "approved", "created", "fulfilled", "unfulfilled", "open", "closed",
}

EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.IGNORECASE)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _name_score(col_name: str, concept: dict) -> float:
    """Proximité entre le nom de colonne et les synonymes du concept."""
    norm = normalize_name(col_name).replace("_", " ")
    tokens = set(norm.split())
    best = 0.0

    for syn in concept["synonyms"]:
        s = syn.replace("_", " ")
        if norm == s:
            # Nom EXACT : « Status » désigne le statut principal, tandis que
            # « Courier Status » ou « Payment Status » désignent des sous-statuts
            # partiels. Le qualificatif restreint la portée de la colonne.
            return 1.15
        s_tokens = set(s.split())
        if s_tokens and s_tokens <= tokens:          # tous les mots présents
            best = max(best, 0.9)
        elif len(s) > 3 and s in norm:               # sous-chaîne significative
            best = max(best, 0.75)
        elif s_tokens & tokens:                      # recouvrement partiel
            inter = len(s_tokens & tokens) / max(len(s_tokens), 1)
            best = max(best, 0.45 + 0.25 * inter)

    for bad in concept.get("penalize", []):
        b = bad.replace("_", " ")
        if b in norm or set(b.split()) <= tokens:
            best -= 0.55
            break

    return max(0.0, min(1.0, best))


def _num(v):
    """Convertit en nombre si possible. Les stats temporelles contiennent du texte."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def _content_score(col_profile: dict, series: pd.Series | None, concept_key: str) -> float:
    """Indices tirés des valeurs elles-mêmes. Renvoie un ajustement dans [-0.4, +0.4]."""
    adj = 0.0
    stats = col_profile.get("stats") or {}
    n_unique = col_profile.get("n_unique", 0)
    ratio = col_profile.get("cardinality_ratio", 0)

    if concept_key == "order_status":
        # Comparaison par SOUS-CHAÎNE : les plateformes utilisent des statuts
        # composés (« Shipped - Delivered to Buyer », « Pending - Waiting for
        # Pick Up »). Une égalité stricte ne les reconnaissait pas, et
        # favorisait une colonne secondaire aux valeurs plus simples.
        tops = [str(v["value"]).strip().lower() for v in col_profile.get("top_values", [])]
        if tops:
            hit = sum(1 for t in tops
                      if any(sv in t for sv in STATUS_VALUES)) / len(tops)
            adj += 0.4 * hit if hit else -0.1

    elif concept_key == "customer_id":
        if series is not None:
            sample = series.dropna().astype(str).head(200)
            if len(sample) and sample.map(lambda x: bool(EMAIL.match(x))).mean() > 0.7:
                adj += 0.35            # une colonne d'e-mails identifie un client
        if 0.05 < ratio < 0.95:
            adj += 0.15                # un client revient : ni unique, ni constant

    elif concept_key == "order_id":
        if ratio > 0.9:
            adj += 0.2

    elif concept_key == "revenue":
        # Les stats d'une colonne temporelle contiennent des dates en texte :
        # on ne compare que si les bornes sont bien numériques.
        mn, mx = _num(stats.get("min")), _num(stats.get("max"))
        if mn is not None and mx is not None:
            if mn >= 0 and mx > 5:
                adj += 0.2
            if 0 <= mx <= 1:
                adj -= 0.35            # un ratio, pas un montant
            if col_profile.get("additive") is False:
                adj -= 0.3

    elif concept_key == "quantity":
        mn, mx = _num(stats.get("min")), _num(stats.get("max"))
        med = _num(stats.get("median"))
        if mn is not None and mx is not None:
            if 0 <= mn and mx <= 1000 and (med is not None and med == int(med)):
                adj += 0.25
            if mx > 10000:
                adj -= 0.25

    elif concept_key == "product_category":
        if 2 <= n_unique <= 100:
            adj += 0.2
        elif n_unique > 500:
            adj -= 0.25

    elif concept_key == "region":
        if 2 <= n_unique <= 300:
            adj += 0.15

    return max(-0.4, min(0.4, adj))


# --------------------------------------------------------------------------

@dataclass
class Candidate:
    column: str
    concept: str
    score: float
    name_score: float
    content_score: float
    structural_ok: bool
    reasons: list = field(default_factory=list)


def _candidates(profile: dict, df: pd.DataFrame | None) -> list[Candidate]:
    out: list[Candidate] = []
    for col in profile["columns"]:
        role = col["role_candidate"]
        if role in {"empty"}:
            continue
        series = df[col["name"]] if df is not None and col["name"] in df.columns else None

        for key, concept in CONCEPTS.items():
            ns = _name_score(col["name"], concept)
            struct_ok = role in concept["structural_roles"]
            cs = _content_score(col, series, key)

            # Le nom seul ne suffit jamais : une colonne nommée « prix » mais
            # remplie de texte n'est pas un montant. Le filtre structurel
            # écarte ces cas au lieu de produire un mapping absurde.
            if not struct_ok:
                if ns < 0.85:
                    continue
                score = ns * 0.45 + cs      # nom très explicite : on garde, dégradé
            else:
                score = ns * 0.7 + cs + 0.15

            # Seuil d'attribution. Fixé à 0,45 après constat : à 0,25, des noms
            # arbitraires (« who », « quoi ») recevaient un rôle métier au hasard
            # avec une confiance de 0,30. Un mapping faux appliqué silencieusement
            # est pire qu'un rôle vide signalé — le commerçant analyserait ses
            # ventes avec la mauvaise colonne sans jamais le savoir.
            if score < 0.45:
                continue

            reasons = []
            if ns >= 0.9:
                reasons.append("nom explicite")
            elif ns >= 0.5:
                reasons.append("nom proche")
            if cs > 0.15:
                reasons.append("contenu cohérent")
            if cs < -0.1:
                reasons.append("contenu douteux")
            if not struct_ok:
                reasons.append(f"type inattendu ({role})")

            out.append(Candidate(col["name"], key, round(min(1.0, score), 3),
                                 round(ns, 3), round(cs, 3), struct_ok, reasons))
    return sorted(out, key=lambda c: -c.score)


def build_mapping(profile: dict, df: pd.DataFrame | None = None) -> dict:
    """
    Propose un mapping colonne → rôle métier.

    Résolution gloutonne : le meilleur couple (colonne, concept) est attribué
    d'abord, puis retiré des candidats. Évite qu'une même colonne serve deux
    rôles, ou que deux colonnes se disputent le même.
    """
    cands = _candidates(profile, df)

    mapping: dict[str, str | None] = {}
    details: dict[str, dict] = {}
    used_cols: set[str] = set()

    for c in cands:
        if c.concept in mapping or c.column in used_cols:
            continue
        mapping[c.concept] = c.column
        used_cols.add(c.column)
        details[c.concept] = {
            "column": c.column,
            "confidence": c.score,
            "name_score": c.name_score,
            "content_score": c.content_score,
            "structural_ok": c.structural_ok,
            "reasons": c.reasons,
            "alternatives": [
                {"column": a.column, "confidence": a.score}
                for a in cands
                if a.concept == c.concept and a.column != c.column
            ][:3],
        }

    # --- Désambiguïsation prix unitaire / montant total -------------------
    # Un nom générique comme « Price » ou « Prix » peut désigner l'un ou
    # l'autre. Le contexte tranche : SI une colonne de quantité existe et
    # qu'aucun prix unitaire n'a été identifié, alors une colonne « prix »
    # au nom non explicite est presque toujours UNITAIRE.
    #
    # Sans cette règle, le chiffre d'affaires d'Online Retail II — l'un des
    # jeux de données e-commerce les plus utilisés — serait la somme des prix
    # unitaires, sans tenir compte des quantités. Erreur massive et silencieuse.
    NOMS_TOTAL = {"total", "montant", "amount", "revenue", "ca", "chiffre",
                  "somme", "value", "subtotal", "net"}

    rev_col = mapping.get("revenue")
    qte_col = mapping.get("quantity")

    # La quantité doit être crédible avant de servir d'argument. Le critère
    # n'est PAS le nombre de valeurs distinctes — une vente en gros peut aller
    # de 1 à 80 000 unités, soit des centaines de valeurs (Online Retail II en
    # compte 722). Le vrai discriminant est que **une quantité est entière**,
    # et que son nom n'évoque pas un identifiant (order_item_id chez Olist est
    # un numéro de séquence d'article, pas une quantité).
    qte_fiable = False
    if qte_col and df is not None and qte_col in df.columns:
        serie = pd.to_numeric(df[qte_col], errors="coerce").dropna()
        entiere = bool(len(serie) and (serie % 1 == 0).mean() > 0.98)
        pas_un_id = not (_tokens(qte_col) & {"id", "identifiant", "numero",
                                             "sequence", "rang", "index"})
        conf = details.get("quantity", {}).get("confidence", 0)
        qte_fiable = entiere and pas_un_id and conf >= 0.6
    elif qte_col:
        info_q = next((c for c in profile["columns"] if c["name"] == qte_col), None)
        qte_fiable = bool(
            info_q and info_q["n_unique"] <= 200
            and not (_tokens(qte_col) & {"id", "identifiant", "numero", "sequence"})
            and details.get("quantity", {}).get("confidence", 0) >= 0.6)

    if rev_col and qte_fiable and not mapping.get("unit_price"):
        toks = _tokens(rev_col)
        explicite_total = bool(toks & NOMS_TOTAL)
        if not explicite_total:
            # « Price », « Prix », « Prix article » → prix unitaire
            mapping["unit_price"] = rev_col
            mapping["revenue"] = None
            d = details.pop("revenue", None)
            if d:
                d["reasons"] = d.get("reasons", []) + [
                    "requalifié en prix unitaire (une quantité existe par ailleurs)"]
                details["unit_price"] = d

    for key in CONCEPTS:
        mapping.setdefault(key, None)

    # --- Repli structurel -------------------------------------------------
    # Quand aucun nom ne ressemble à un synonyme connu, le rôle structurel peut
    # trancher : s'il n'existe qu'une seule colonne temporelle et qu'aucune date
    # n'est mappée, c'est forcément elle. Indispensable pour les schémas
    # imprévisibles — « quand », « Name », ou tout export non standard.
    # Confiance délibérément basse : l'utilisateur devra confirmer.
    libres = [c for c in profile["columns"]
              if c["name"] not in used_cols
              and c["role_candidate"] not in {"empty", "constant"}]

    def _repli(concept: str, roles: set[str], conf: float, motif: str,
               unique_seulement: bool = True):
        if mapping.get(concept):
            return
        cands_r = [c for c in libres if c["role_candidate"] in roles
                   and c["name"] not in used_cols]
        if not cands_r or (unique_seulement and len(cands_r) > 1):
            return
        c = cands_r[0]
        mapping[concept] = c["name"]
        used_cols.add(c["name"])
        details[concept] = {
            "column": c["name"], "confidence": conf,
            "name_score": 0.0, "content_score": 0.0,
            "structural_ok": True, "reasons": [motif, "à confirmer"],
            "alternatives": [],
        }

    _repli("order_date", {"temporal"}, 0.45, "seule colonne de type date")
    _repli("revenue", {"measure"}, 0.40, "seule colonne de montant")
    _repli("order_id", {"identifier"}, 0.35, "identifiant sans nom explicite")
    # --------------------------------------------------------------------

    # Colonnes numériques non attribuées : mesures secondaires exploitables
    extras = [c["name"] for c in profile["columns"]
              if c["role_candidate"] == "measure" and c["name"] not in used_cols]

    manquants = [k for k in REQUIRED if not mapping.get(k)]
    faibles = [k for k, d in details.items() if d["confidence"] < 0.6]

    # Pistes écartées faute de confiance suffisante. Elles ne sont PAS appliquées,
    # mais proposées comme suggestions dans les questions posées à l'utilisateur.
    suggestions: dict[str, list] = {}
    for c in cands:
        if mapping.get(c.concept) or c.column in used_cols:
            continue
        suggestions.setdefault(c.concept, []).append(
            {"column": c.column, "confidence": c.score})

    return {
        "mapping": mapping,
        "details": details,
        "weak_suggestions": {k: v[:3] for k, v in suggestions.items()},
        "unmapped_columns": [c["name"] for c in profile["columns"]
                             if c["name"] not in used_cols
                             and c["role_candidate"] not in {"empty", "constant"}],
        "additional_measures": extras,
        "missing_required": manquants,
        "low_confidence": faibles,
        "needs_review": bool(manquants or faibles),
        "available_analyses": available_analyses(mapping),
    }


def available_analyses(mapping: dict) -> dict:
    """Quelles analyses sont possibles avec ce mapping — et lesquelles ne le sont pas."""
    m = {k: v for k, v in mapping.items() if v}
    besoins = {
        "revenue_over_time":  ("revenue", "order_date"),
        "top_products":       ("revenue", "product_id"),
        "category_mix":       ("revenue", "product_category"),
        "basket_size":        ("revenue", "order_id", "order_date"),
        "customer_retention": ("customer_id", "order_date"),
        "rfm_segmentation":   ("customer_id", "order_date", "revenue"),
        "geographic_split":   ("revenue", "region"),
        "weekday_pattern":    ("revenue", "order_date"),
        "payment_analysis":   ("revenue", "payment_method"),
        "channel_analysis":   ("revenue", "channel"),
    }
    dispo, indispo = [], []
    for nom, req in besoins.items():
        absents = [r for r in req if r not in m]
        if absents:
            indispo.append({"analysis": nom,
                            "missing": [CONCEPTS[a]["label"] for a in absents]})
        else:
            dispo.append(nom)
    return {"available": dispo, "unavailable": indispo}


def mapping_questions(result: dict) -> list[dict]:
    """
    Questions à poser à l'utilisateur, en langage clair.

    On ne demande que ce qui est incertain ou manquant — un mapping sûr
    est simplement affiché pour confirmation.
    """
    out = []
    for key in result["missing_required"]:
        out.append({
            "concept": key,
            "severity": "high",
            "question": f"Quelle colonne contient {CONCEPTS[key]['label'].lower()} ?",
            "explanation": "Sans cette information, l'analyse ne peut pas être faite.",
            "suggestions": result["unmapped_columns"][:6],
        })
    for key in result["low_confidence"]:
        d = result["details"][key]
        out.append({
            "concept": key,
            "severity": "medium",
            "question": f"« {d['column']} » correspond-elle bien à "
                        f"{CONCEPTS[key]['label'].lower()} ?",
            "explanation": "Nous n'en sommes pas certains, merci de vérifier.",
            "suggestions": [a["column"] for a in d["alternatives"]] or
                           result["unmapped_columns"][:4],
        })
    return out


def to_simple(result: dict) -> dict:
    """Mapping épuré {concept: colonne}, prêt pour diagnose() et Cleaner."""
    return {k: v for k, v in result["mapping"].items() if v}


# --------------------------------------------------------------------------
# Colonnes dérivées
# --------------------------------------------------------------------------

def derivations_possibles(mapping: dict, profile: dict) -> list[dict]:
    """
    Colonnes calculables à partir de celles qui existent.

    Beaucoup de fichiers réels ne contiennent PAS de montant total : seulement
    une quantité et un prix unitaire. C'est le cas d'Online Retail II, l'un des
    jeux de données e-commerce les plus utilisés. Sans dérivation, le pipeline
    s'arrête sur « aucune colonne de montant trouvée » alors que l'information
    est présente — simplement répartie sur deux colonnes.

    La dérivation n'est jamais appliquée en silence : elle est proposée, et
    l'utilisateur valide. Créer une colonne revient à produire une donnée qui
    n'était pas dans le fichier.
    """
    out = []
    m = {k: v for k, v in mapping.items() if v}

    if not m.get("revenue") and m.get("quantity") and m.get("unit_price"):
        out.append({
            "cible": "revenue",
            "nom_colonne": "Montant (calculé)",
            "formule": "quantity * unit_price",
            "sources": [m["quantity"], m["unit_price"]],
            "user_message": "Votre fichier ne contient pas de colonne « montant ».",
            "user_explanation": f"Nous pouvons la calculer en multipliant "
                                f"« {m['quantity']} » par « {m['unit_price']} ».",
            "recommandation": True,
        })

    if not m.get("revenue") and m.get("order_total"):
        out.append({
            "cible": "revenue",
            "nom_colonne": m["order_total"],
            "formule": "alias",
            "sources": [m["order_total"]],
            "user_message": "Nous utiliserons le total de commande comme montant.",
            "user_explanation": f"La colonne « {m['order_total']} » servira de "
                                f"référence pour votre chiffre d'affaires.",
            "recommandation": True,
        })

    return out


def appliquer_derivations(df, mapping: dict, derivations: list[dict],
                          acceptees: list[str] | None = None):
    """
    Crée les colonnes dérivées acceptées et met le mapping à jour.

    Retourne (df modifié, mapping modifié, journal des créations).
    """
    import pandas as pd
    from .profiler import clean_numeric_strings

    acceptees = acceptees if acceptees is not None else [d["cible"] for d in derivations]
    df = df.copy()
    mapping = dict(mapping)
    journal = []

    for d in derivations:
        if d["cible"] not in acceptees:
            continue

        if d["formule"] == "quantity * unit_price":
            qte = pd.to_numeric(clean_numeric_strings(df[d["sources"][0]]), errors="coerce")
            pu = pd.to_numeric(clean_numeric_strings(df[d["sources"][1]]), errors="coerce")
            df[d["nom_colonne"]] = (qte * pu).round(2)
        elif d["formule"] == "alias":
            df[d["nom_colonne"]] = pd.to_numeric(
                clean_numeric_strings(df[d["sources"][0]]), errors="coerce")
        else:
            continue

        mapping[d["cible"]] = d["nom_colonne"]
        journal.append({"colonne_creee": d["nom_colonne"], "formule": d["formule"],
                        "sources": d["sources"]})

    return df, mapping, journal


def remap_apres_nettoyage(mapping: dict, df_clean, profile_clean: dict) -> dict:
    """
    Conserve le mapping validé après nettoyage, au lieu de le recalculer.

    Re-lancer `build_mapping` sur les données nettoyées est une erreur : le
    mapping a pu être corrigé par l'utilisateur, et une colonne dérivée
    (« Montant (calculé) ») n'a aucune raison d'être redécouverte à
    l'identique. Recalculer, c'est perdre une décision humaine.

    On se contente d'écarter les rôles dont la colonne a disparu au nettoyage,
    et de compléter les rôles vides avec ce que le profil permet d'identifier.
    """
    conserve = {k: v for k, v in mapping.items()
                if v and v in df_clean.columns}

    # Complément uniquement pour les rôles restés vides
    manquants = [k for k in CONCEPTS if k not in conserve]
    if manquants:
        propose = build_mapping(profile_clean, df_clean)["mapping"]
        deja = set(conserve.values())
        for k in manquants:
            v = propose.get(k)
            if v and v in df_clean.columns and v not in deja:
                conserve[k] = v
                deja.add(v)

    return conserve
