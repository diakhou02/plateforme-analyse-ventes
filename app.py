"""
Plateforme d'analyse pour commerçants en ligne.

    streamlit run app.py

Parcours en quatre écrans :
    1. Dépôt du fichier
    2. Diagnostic qualité et arbitrages
    3. Vérification du mapping métier
    4. Analyses, graphiques et interprétations

Aucune logique métier ici : tout vient de src/. L'application n'est qu'une
couche de présentation — c'est ce qui garantit que ce qui a été validé dans
les notebooks est exactement ce qui s'exécute en production.
"""

from __future__ import annotations

import io
import json
import traceback
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.charts import build_figure
from src.cleaner import Cleaner, controle_vraisemblance
from src.executor import run_plan
from src.interpreter import build_synthesis, interpret_one
from src.mapper import (CONCEPTS, appliquer_derivations, build_mapping,
                        derivations_possibles, mapping_questions,
                        remap_apres_nettoyage, to_simple)
from src.planner import guaranteed_specs, make_plan
from src.profiler import build_profile
from src.quality import diagnose
from src.sector import detecter_secteur
from src.reader import read_any
from src.validator import validate_plan

MODELE_DEFAUT = os.environ.get("MODELE", "gemini-3.5-flash-lite")

st.set_page_config(page_title="Analyse de vos ventes", page_icon="📊",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1080px;}
  h1 {font-size: 1.9rem !important;}
  .carte {background:#f8fafc; border-left:3px solid #2563eb;
          padding:16px 20px; border-radius:4px; margin-bottom:18px;}
  .carte-alerte {background:#fef2f2; border-left-color:#dc2626;}
  .carte-ok {background:#f0fdf4; border-left-color:#059669;}
  .etiquette {font-size:.78rem; color:#64748b; text-transform:uppercase;
              letter-spacing:.04em; margin-bottom:6px;}
  .score {font-size:3.2rem; font-weight:600; line-height:1;}
  div[data-testid="stMetricValue"] {font-size:1.5rem;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# État
# --------------------------------------------------------------------------

def init():
    for cle, val in {
        "etape": 1, "df_raw": None, "rapport_lecture": None, "profil": None,
        "map_res": None, "mapping": None, "rapport": None, "cleaner": None,
        "df_clean": None, "facts": None, "interpretations": None,
        "synthese": None, "decisions": {}, "secteur": None,
        "fichier_courant": None,
    }.items():
        st.session_state.setdefault(cle, val)


def aller(n: int):
    st.session_state.etape = n


def cle_api() -> str | None:
    """Secret Streamlit en priorité, variable d'environnement en repli."""
    try:
        if "LLM_API_KEY" in st.secrets:
            return st.secrets["LLM_API_KEY"]
    except Exception:
        pass
    return os.environ.get("LLM_API_KEY")


init()


# --------------------------------------------------------------------------
# En-tête
# --------------------------------------------------------------------------

ETAPES = ["Votre fichier", "Vérification", "Vos colonnes", "Vos résultats"]
cols = st.columns([3, 5])
with cols[0]:
    st.markdown("## 📊 Analyse de vos ventes")
with cols[1]:
    actuelle = st.session_state.etape
    st.markdown(
        "<div style='padding-top:14px;font-size:.9rem'>"
        + "  →  ".join(
            f"<b style='color:#2563eb'>{n}. {e}</b>" if i + 1 == actuelle
            else f"<span style='color:#cbd5e1'>{n}. {e}</span>"
            for i, (n, e) in enumerate(zip(range(1, 5), ETAPES)))
        + "</div>", unsafe_allow_html=True)
st.divider()


# ==========================================================================
# ÉTAPE 1 — Dépôt du fichier
# ==========================================================================

if st.session_state.etape == 1:
    st.markdown("### Déposez votre fichier de ventes")
    st.write("Un export de votre boutique en ligne, ou un tableau Excel. "
             "Nous nous chargeons du reste.")

    fichier = st.file_uploader("Fichier de ventes",
                               type=["csv", "xlsx", "xls", "tsv"],
                               label_visibility="collapsed")

    with st.expander("Ce qui est attendu dans votre fichier"):
        st.write("**Indispensable** — une colonne de date et une colonne de montant.")
        st.write("**Utile** — numéro de commande, client, produit, catégorie, "
                 "ville, statut, quantité.")
        st.caption("Les noms de colonnes n'ont pas d'importance : ils sont "
                   "reconnus automatiquement, en français comme en anglais.")

    if fichier is not None:
        # Un nouveau fichier remet TOUT à zéro. Sans cela, les décisions, le
        # secteur et les analyses du fichier précédent survivent dans la
        # session et s'appliquent à des données auxquelles ils ne
        # correspondent plus.
        signature = f"{fichier.name}:{fichier.size}"
        if st.session_state.get("fichier_courant") != signature:
            for cle in ("decisions", "secteur", "facts", "interpretations",
                        "synthese", "cleaner", "df_clean", "mapping",
                        "map_res", "rapport", "profil"):
                st.session_state[cle] = {} if cle == "decisions" else None
            st.session_state["fichier_courant"] = signature

        try:
            with st.spinner("Lecture du fichier…"):
                df, rap_lecture = read_any(fichier.getvalue(), filename=fichier.name)
                profil = build_profile(df, rap_lecture)
                map_res = build_mapping(profil, df)
                mapping = to_simple(map_res)
                rapport = diagnose(df, profil, rap_lecture, mapping)
                rapport["_profile_columns"] = profil["columns"]

            st.session_state.update(
                df_raw=df, rapport_lecture=rap_lecture, profil=profil,
                map_res=map_res, mapping=mapping, rapport=rapport)

            c1, c2, c3 = st.columns(3)
            c1.metric("Lignes", f"{len(df):,}".replace(",", " "))
            c2.metric("Colonnes", df.shape[1])
            c3.metric("Qualité", f"{rapport['quality_score']}/100")

            if rapport["blocked"]:
                for b in rapport["blocked"]:
                    st.error(f"**{b['user_message']}**\n\n{b['user_explanation']}")
            else:
                st.dataframe(df.head(5), use_container_width=True)
                st.button("Continuer  →", type="primary",
                          on_click=aller, args=(2,))

        except ValueError as e:
            # B04 est le SEUL cas où le fichier est réellement illisible.
            # Attraper toute ValueError sous ce message masquait les erreurs
            # survenues plus loin dans le pipeline — profilage, mapping,
            # diagnostic — et rendait tout diagnostic impossible.
            if "B04" in str(e):
                st.error("Ce fichier n'a pas pu être lu. Essayez de le "
                         "réenregistrer au format CSV depuis votre tableur.")
            else:
                st.error("Une erreur est survenue pendant l'analyse de votre fichier.")
                with st.expander("Détail technique"):
                    st.code(f"{type(e).__name__}: {e}")
                    st.code(traceback.format_exc())
        except Exception as e:
            st.error("Une erreur est survenue pendant l'analyse de votre fichier.")
            with st.expander("Détail technique"):
                st.code(f"{type(e).__name__}: {e}")
                st.code(traceback.format_exc())


# ==========================================================================
# ÉTAPE 2 — Diagnostic et arbitrages
# ==========================================================================

elif st.session_state.etape == 2:
    rapport = st.session_state.rapport
    score = rapport["quality_score"]
    couleur = "#059669" if score >= 75 else ("#f59e0b" if score >= 50 else "#dc2626")

    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown(f"<div class='score' style='color:{couleur}'>{score}"
                    f"<span style='font-size:1.1rem;color:#94a3b8'>/100</span></div>",
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f"### {rapport['quality_label']}")
        st.caption(f"{rapport['n_issues']} points examinés · "
                   f"{rapport['n_decisions_required']} décision(s) à prendre")

    st.divider()

    decisions = rapport["decisions_required"]

    if decisions:
        st.markdown("#### Quelques points à vérifier")
        st.caption("Nos recommandations sont pré-sélectionnées. "
                   "Vous pouvez les modifier.")

        for n_issue, issue in enumerate(decisions):
            # La clé doit être unique : une même anomalie peut toucher
            # plusieurs colonnes (C04 sur « Discount » ET « Region »), et
            # deux widgets de même clé font planter Streamlit.
            iid = issue["id"]
            cle = f"{iid}_{n_issue}_{'_'.join(issue.get('columns') or [])}"
            with st.container(border=True):
                st.markdown(f"**{issue['user_message']}**")
                st.write(issue["user_explanation"])
                if issue.get("impact_if_ignored"):
                    st.warning(issue["impact_if_ignored"], icon="⚠️")

                options = issue.get("options", [])
                labels = [o["label"] for o in options]
                defaut = next((i for i, o in enumerate(options)
                               if o.get("is_default")), 0)
                choix = st.radio("Que faire ?", labels, index=defaut,
                                 key=f"radio_{cle}", horizontal=True,
                                 label_visibility="collapsed")
                action = options[labels.index(choix)]["action"]
                st.session_state.decisions[iid] = action

                if action == "preview" and issue.get("sample_rows"):
                    idx = [i for i in issue["sample_rows"]
                           if i in st.session_state.df_raw.index]
                    if idx:
                        st.dataframe(st.session_state.df_raw.loc[idx],
                                     use_container_width=True)
    else:
        st.success("Aucune correction n'est nécessaire.", icon="✅")

    auto = rapport["auto_applied"] + rapport["auto_notified"]
    if auto:
        with st.expander(f"Corrections déjà appliquées ({len(auto)})"):
            for i in auto:
                st.markdown(f"- **{i['user_message']}**  \n"
                            f"  <span style='color:#64748b'>{i['user_explanation']}"
                            f"</span>", unsafe_allow_html=True)

    st.divider()
    c1, c2 = st.columns([1, 4])
    with c1:
        st.button("←  Retour", on_click=aller, args=(1,))
    with c2:
        if st.button("Corriger et continuer  →", type="primary"):
            nettoyeur = Cleaner(st.session_state.df_raw, rapport,
                                st.session_state.mapping)
            nettoyeur.apply_defaults()
            connues = {i["id"] for i in rapport.get("issues", [])}
            for iid, action in st.session_state.decisions.items():
                if action != "preview" and iid in connues:
                    nettoyeur.set_decision(iid, action)
            st.session_state.cleaner = nettoyeur
            st.session_state.df_clean = nettoyeur.result()
            aller(3)
            st.rerun()


# ==========================================================================
# ÉTAPE 3 — Vérification du mapping
# ==========================================================================

elif st.session_state.etape == 3:
    df_clean = st.session_state.df_clean
    nettoyeur = st.session_state.cleaner

    controle = controle_vraisemblance(nettoyeur, st.session_state.mapping)

    st.markdown("### Votre fichier est prêt")

    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes analysables", f"{len(df_clean):,}".replace(",", " "),
              delta=f"{len(df_clean) - len(st.session_state.df_raw)}")
    rev = st.session_state.mapping.get("revenue")
    if rev and rev in df_clean.columns:
        ca = pd.to_numeric(df_clean[rev], errors="coerce").sum()
        eca = controle["ecarts"].get("chiffre_affaires", {})
        c2.metric("Chiffre d'affaires", f"{ca:,.0f}".replace(",", " "),
                  delta=f"{eca.get('variation_pct', 0):.1f} %")
    cust = st.session_state.mapping.get("customer_id")
    if cust and cust in df_clean.columns:
        c3.metric("Clients", f"{df_clean[cust].nunique():,}".replace(",", " "))

    # Le système montre l'ampleur de son intervention avant validation
    if controle["alertes"]:
        for a in controle["alertes"]:
            st.warning(a["message"], icon="⚠️")
    elif controle["statut"] == "modifie":
        st.info(controle["resume"], icon="ℹ️")

    # Colonnes calculables : proposées, jamais créées en silence
    derivs = derivations_possibles(st.session_state.mapping,
                                   build_profile(df_clean))
    acceptees = []
    if derivs:
        st.divider()
        st.markdown("#### Une colonne peut être calculée")
        for d in derivs:
            with st.container(border=True):
                st.markdown(f"**{d['user_message']}**")
                st.write(d["user_explanation"])
                if st.checkbox("Calculer cette colonne", value=True,
                               key=f"deriv_{d['cible']}"):
                    acceptees.append(d["cible"])
        if acceptees:
            df_clean, nouveau_mapping, journal = appliquer_derivations(
                df_clean, st.session_state.mapping, derivs, acceptees)
            st.session_state.df_clean = df_clean
            st.session_state.mapping = nouveau_mapping

    # Secteur déduit des noms de produits — affiché, jamais imposé
    if st.session_state.secteur is None:
        with st.spinner("Identification de votre activité…"):
            st.session_state.secteur = detecter_secteur(
                df_clean, st.session_state.mapping,
                model=MODELE_DEFAUT, api_key=cle_api(), log_dir=None)

    sect = st.session_state.secteur
    st.divider()
    st.markdown("#### Votre activité")

    from src.sector import SECTEURS
    libelles = ["— non précisé —"] + [v["label"] for v in SECTEURS.values()]
    cles = [None] + list(SECTEURS)
    idx = cles.index(sect["secteur"]) if sect.get("secteur") in cles else 0

    c1, c2 = st.columns([2, 3])
    with c1:
        choix = st.selectbox("Secteur d'activité", libelles, index=idx,
                             label_visibility="collapsed")
        nouvelle_cle = cles[libelles.index(choix)]
        if nouvelle_cle != sect.get("secteur"):
            from src.sector import SECTEUR_INCONNU
            ctx = SECTEURS.get(nouvelle_cle, SECTEUR_INCONNU)
            st.session_state.secteur = {**sect, "secteur": nouvelle_cle,
                                        "label": ctx["label"],
                                        "saisonnalite": ctx["saisonnalite"],
                                        "enjeux": ctx["enjeux"],
                                        "leviers": ctx.get("leviers", []),
                                        "confiance": 1.0, "a_confirmer": False,
                                        "methode": "choisi par l'utilisateur"}
            sect = st.session_state.secteur
    with c2:
        if sect.get("indices"):
            st.caption("Déduit de vos produits : "
                       + ", ".join(sect["indices"][:5]))
        if sect.get("saisonnalite"):
            st.caption(sect["saisonnalite"])

    st.divider()
    st.markdown("#### Vos colonnes")
    st.caption("Voici comment nous avons compris votre fichier. "
               "Corrigez si nécessaire.")

    dispo = ["— aucune —"] + list(df_clean.columns)
    mapping_final = {}

    cles = [k for k in CONCEPTS if st.session_state.mapping.get(k)
            or CONCEPTS[k]["required"]]
    autres = [k for k in CONCEPTS if k not in cles]

    for groupe, titre in [(cles, None), (autres, "Autres colonnes (facultatif)")]:
        if titre:
            groupe_ouvert = st.expander(titre)
            conteneur = groupe_ouvert
        else:
            conteneur = st.container()
        with conteneur:
            for k in groupe:
                actuel = st.session_state.mapping.get(k)
                idx = dispo.index(actuel) if actuel in dispo else 0
                detail = (st.session_state.map_res.get("details") or {}).get(k, {})
                conf = detail.get("confidence")
                aide = None
                if conf is not None and conf < 0.6:
                    aide = "Nous ne sommes pas certains — merci de vérifier."
                label = CONCEPTS[k]["label"]
                if CONCEPTS[k]["required"]:
                    label += " *"
                choix = st.selectbox(label, dispo, index=idx,
                                     key=f"map_{k}", help=aide)
                if choix != "— aucune —":
                    mapping_final[k] = choix

    manquants = [k for k in CONCEPTS if CONCEPTS[k]["required"]
                 and k not in mapping_final]
    if manquants:
        st.error("Indiquez : " + ", ".join(CONCEPTS[k]["label"].lower()
                                           for k in manquants))

    st.divider()
    c1, c2 = st.columns([1, 4])
    with c1:
        st.button("←  Retour", on_click=aller, args=(2,))
    with c2:
        if st.button("Analyser mes ventes  →", type="primary",
                     disabled=bool(manquants)):
            st.session_state.mapping = mapping_final
            aller(4)
            st.rerun()


# ==========================================================================
# ÉTAPE 4 — Résultats
# ==========================================================================

elif st.session_state.etape == 4:
    df_clean = st.session_state.df_clean
    mapping = st.session_state.mapping
    api_key = cle_api()

    if st.session_state.facts is None:
        with st.spinner("Analyse de vos ventes en cours…"):
            profil = build_profile(df_clean)

            # Le mapping validé par l'utilisateur est CONSERVÉ, jamais
            # recalculé : le recalculer effacerait ses corrections et ne
            # retrouverait pas les colonnes dérivées.
            mapping = remap_apres_nettoyage(mapping, df_clean, profil)
            st.session_state.mapping = mapping

            if api_key:
                from src.planner import build_llm_profile
                map_res = build_mapping(profil, df_clean)
                lp = build_llm_profile(profil, map_res, st.session_state.rapport)
                plan = make_plan(lp, mapping, model=MODELE_DEFAUT, api_key=api_key)
            else:
                plan = {"specs": guaranteed_specs(mapping)}

            res = validate_plan(plan, profil, df_clean,
                                max_specs=8, min_llm_specs=3)
            facts, _ = run_plan(res["specs"], df_clean, mapping, unite="")
            st.session_state.facts = facts

            interpretations = {}
            if api_key:
                for f in facts:
                    interpretations[f["spec_id"]] = interpret_one(
                        f, model=MODELE_DEFAUT, api_key=api_key,
                        secteur=st.session_state.secteur)
                st.session_state.synthese = build_synthesis(
                    facts, model=MODELE_DEFAUT, api_key=api_key,
                    secteur=st.session_state.secteur)
            st.session_state.interpretations = interpretations

    facts = st.session_state.facts
    interpretations = st.session_state.interpretations or {}
    synthese = st.session_state.synthese

    # --- Synthèse en tête : ce que le commerçant lit en premier ---
    sect = st.session_state.secteur
    if sect and sect.get("label"):
        st.caption(f"Analyse adaptée à votre activité : **{sect['label']}**")

    if synthese and not synthese.get("error"):
        st.markdown("### En résumé")
        st.markdown(f"<div class='carte'>{synthese['situation']}</div>",
                    unsafe_allow_html=True)

        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**Vos priorités**")
            for n, p in enumerate(synthese.get("priorites", []), 1):
                st.markdown(f"**{n}. {p['titre']}**  \n"
                            f"<span style='color:#475569'>{p['explication']}</span>",
                            unsafe_allow_html=True)
        with c2:
            st.markdown("**À faire cette semaine**")
            st.markdown(f"<div class='carte carte-ok'>"
                        f"{synthese.get('premiere_action', '')}</div>",
                        unsafe_allow_html=True)
        st.divider()

    if not api_key:
        st.info("Les commentaires automatiques ne sont pas activés : "
                "seuls les graphiques sont affichés.", icon="ℹ️")

    # --- Une analyse = un graphique PUIS son commentaire ---
    for f in facts:
        st.markdown(f"#### {f['title']}")
        if f.get("business_question"):
            st.caption(f["business_question"])

        st.plotly_chart(build_figure(f), use_container_width=True,
                        key=f["spec_id"])

        interp = interpretations.get(f["spec_id"], {})
        if interp and not interp.get("error"):
            st.markdown(
                f"<div class='carte'>"
                f"<div class='etiquette'>Ce que montre ce graphique</div>"
                f"{interp.get('constat', '')}"
                f"<div class='etiquette' style='margin-top:14px'>"
                f"Où est le problème</div>{interp.get('diagnostic', '')}"
                f"</div>", unsafe_allow_html=True)
            if interp.get("actions"):
                st.markdown("**Ce que vous pouvez faire**")
                for a in interp["actions"]:
                    st.markdown(f"- {a}")

        cov = f.get("coverage", {})
        if cov.get("n_lignes_exclues"):
            st.caption(f"Calculé sur {cov['n_lignes_utilisees']} lignes "
                       f"({cov['n_lignes_exclues']} sans information exploitable).")
        st.divider()

    # --- Export ---
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.button("←  Retour", on_click=aller, args=(3,))
    with c2:
        csv = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger les données nettoyées", csv,
                           "ventes_nettoyees.csv", "text/csv")
    with c3:
        rapport_export = {
            "synthese": synthese,
            "analyses": [{"titre": f["title"],
                          "chiffres": f["computed_stats"],
                          "problemes": [p["pattern_id"]
                                        for p in f["detected_patterns"]],
                          "commentaire": interpretations.get(f["spec_id"], {})}
                         for f in facts],
        }
        st.download_button(
            "Télécharger le rapport",
            json.dumps(rapport_export, ensure_ascii=False, indent=2,
                       default=str).encode("utf-8"),
            "rapport_analyse.json", "application/json")


st.divider()
st.caption("Tous les chiffres affichés sont calculés directement depuis votre "
           "fichier. Les commentaires ne peuvent citer que ces chiffres.")
