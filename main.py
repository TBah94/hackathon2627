"""
Application Streamlit — Projet 4 : Modélisation des Séries Temporelles
======================================================================
Synthèse exécutive & modélisation sur données réelles NFL :
- Cas NFL : Détection d'impacts & profil biomécanique (GARCH).
- Cas NFL : Cinématique & modélisation de vitesse (ARIMA).
- Cas NFL : Volatilité réactive immédiate sans mémoire (ARCH).
- GARCH(1,1) : Validation par simulation (Pr. Abass Sagna).
- ARCH(1) : Validation par simulation (Pr. Abass Sagna).
- ARIMA(p,d,q) : Validation par simulation (Pr. Abass Sagna).
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Imports directs des modules du projet
import garch
import arch_module
import arima

st.set_page_config(
    page_title="Hackathon — Séries Temporelles NFL",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement mutualisé des données NFL en cache
@st.cache_data(show_spinner=True)
def load_cached_nfl_data():
    tracking_file = (
        os.path.join("data", "game_tracking_2025.csv")
        if os.path.exists(os.path.join("data", "game_tracking_2025.csv"))
        else os.path.join("data", "games_tracking_2025.csv")
    )
    combine_file = os.path.join("data", "combine_results.csv")
    players_file = os.path.join("data", "players.csv")
    
    if not (os.path.exists(tracking_file) and os.path.exists(combine_file)):
        return None, None, []
    
    df_raw, df_weight = garch.load_nfl_data(tracking_file, combine_file, players_file)
    plays = garch.get_available_plays(df_raw, df_weight)
    return df_raw, df_weight, plays


# ==============================================================================
# EN-TÊTE SYNTHÉTIQUE — DASHBOARD EXÉCUTIF
# ==============================================================================
st.title("🏈 Modélisation des Séries Temporelles — Chocs NFL")
st.markdown(
    """
    **Hackathon 2027 — Projet 4 : Validation des modèles par simulation & application NFL**  
    *Comment séparer l'élan de course de la violence mécanique d'un plaquage à l'aide de séries temporelles.*
    """
)

with st.container():
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info(
            """
            **🔄 ARIMA($p, d, q$)**  
            *Composante :* **Moyenne $\\mu_t$**  
            *Signal :* Vitesse de course $v(t)$ (m/s)  
            *Rôle :* Prédit l'inertie et la cinématique de course libre.
            """
        )
    with col_b:
        st.warning(
            """
            **⚡ ARCH($q$)**  
            *Composante :* **Variance $\\sigma_t^2$ instantanée**  
            *Signal :* Force de freinage $F_{decel}$ (N)  
            *Rôle :* Détecte le choc mécanique net sans mémoire ($\\beta = 0$).
            """
        )
    with col_c:
        st.success(
            """
            **📈 GARCH($p, q$)**  
            *Composante :* **Variance $\\sigma_t^2$ avec mémoire**  
            *Signal :* Force de freinage $F_{decel}$ (N)  
            *Rôle :* Modélise la durée de lutte et la persistance ($\\beta > 0$).
            """
        )

st.divider()

# ==============================================================================
# BARRE LATÉRALE : NAVIGATION & MÉMO DISCRET
# ==============================================================================
st.sidebar.title("Navigation")
mode = st.sidebar.radio(
    "Mode d'analyse :",
    [
        "Cas NFL — Chocs & Freinage (GARCH)",
        "Cas NFL — Volatilité Réactive (ARCH)",
        "Cas NFL — Tendance & Vitesse (ARIMA)",
        "GARCH(1,1) — Simulation théorique",
        "ARCH(1) — Simulation théorique",
        "ARIMA(p,d,q) — Simulation théorique",
    ]
)

st.sidebar.divider()

with st.sidebar.expander("ℹ️ Mémo modèles & équations", expanded=False):
    st.markdown(
        r"""
        **Principe :** $X_t = \mu_t + \sigma_t z_t$
        
        * **ARIMA** — Moyenne $\mu_t$
          * *Signal :* vitesse $v(t)$ (m/s)
          * *Rôle :* inertie et trajectoire de course
        
        * **ARCH(q)** — Variance $\sigma_t^2 = \omega + \sum_{i=1}^q \alpha_i \varepsilon_{t-i}^2$
          * *Signal :* force de freinage $F_{\text{decel}}$ (N)
          * *Rôle :* choc sec immédiat ($\beta = 0$)
        
        * **GARCH(p,q)** — Variance $\sigma_t^2 = \omega + \sum \alpha_i \varepsilon_{t-i}^2 + \sum \beta_j \sigma_{t-j}^2$
          * *Signal :* force de freinage $F_{\text{decel}}$ (N)
          * *Rôle :* persistance / durée de la lutte ($\beta > 0$)
          * *Règle :* $\sum \alpha + \sum \beta < 1$ (stationnarité)
        
        ---
        **Verdict NFL :** $\beta$ non significatif ($p > 0{,}05$) $\implies$ le plaquage est physiquement un **ARCH(1) pur**.
        """
    )


# ==============================================================================
# 1. CAS NFL : ANALYSE DES CHOCS GARCH
# ==============================================================================
if mode == "Cas NFL — Chocs & Freinage (GARCH)":
    st.sidebar.subheader("Ordre du modèle GARCH")
    p_garch = st.sidebar.slider("Ordre GARCH (p) — mémoire de variance", min_value=1, max_value=2, value=1)
    q_garch = st.sidebar.slider("Ordre ARCH (q) — réaction au choc", min_value=1, max_value=2, value=1)

    st.subheader(f"Analyse d'Impact Réel : Modèle GARCH({p_garch}, {q_garch})")
    
    with st.spinner("Chargement et indexation des données NFL..."):
        df_raw, df_weight, plays = load_cached_nfl_data()
    
    if df_raw is None or not plays:
        st.error("Données introuvables. Vérifiez la présence des fichiers dans `data/`.")
    else:
        st.sidebar.subheader("Sélection de l'action")
        
        df_plays = pd.DataFrame(plays, columns=["game_id", "play_id", "nfl_id"])
        df_plays = df_plays.merge(
            df_weight[["nfl_id", "display_name", "player_position", "mass_kg"]],
            on="nfl_id",
            how="left"
        ).fillna({"display_name": "Joueur", "player_position": "N/A"})

        df_plays["label"] = (
            df_plays["display_name"] + " (" + df_plays["player_position"] + ") | Game " + 
            df_plays["game_id"].astype(str) + " | Play " + df_plays["play_id"].astype(str)
        )
        
        selected_label = st.sidebar.selectbox("Action avec contact complet :", df_plays["label"].tolist(), key="garch_play_select")
        selected_row = df_plays[df_plays["label"] == selected_label].iloc[0]

        g_id = int(selected_row["game_id"])
        p_id = int(selected_row["play_id"])
        player_id = int(selected_row["nfl_id"])
        player_name = str(selected_row["display_name"])
        player_pos = str(selected_row["player_position"])
        mass_kg = float(selected_row["mass_kg"])

        with st.spinner(f"Calcul des forces et ajustement GARCH({p_garch}, {q_garch})..."):
            df_play, t_start, t_end, dt_contact = garch.compute_player_kinematics(
                df_raw, mass_kg, g_id, p_id, player_id
            )
            # Dans arch_model : p = ordre ARCH (choc), q = ordre GARCH (mémoire)
            model_res, sigma_series = garch.fit_garch_impact(
                df_play["force_decel_N"], p=q_garch, q=p_garch
            )

        f_max = df_play["force_decel_N"].max()
        sigma_max = sigma_series.max()
        
        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
        kpi1.metric("Joueur", player_name)
        kpi2.metric("Poste", player_pos)
        kpi3.metric("Masse", f"{mass_kg:.0f} kg")
        kpi4.metric("Fenêtre contact", f"{dt_contact:.2f} s")
        kpi5.metric("Force max freinage", f"{f_max:.0f} N")
        kpi6.metric(r"Pic volatilité $\sigma_t$", f"{sigma_max:.0f} N")

        fig_nfl = garch.plot_nfl_analysis(
            df_play, sigma_series, t_start, t_end, dt_contact, 
            player_id, mass_kg, player_name=player_name, player_position=player_pos,
            order=(p_garch, q_garch)
        )
        st.pyplot(fig_nfl)

        df_params = pd.DataFrame({
            "Paramètre": model_res.params.index,
            "Valeur": model_res.params.values.round(4),
            "Écart-type": model_res.std_err.values.round(4),
            "p-value": model_res.pvalues.values.round(4),
        })

        alpha_sum = sum(val for name, val in model_res.params.items() if name.startswith("alpha"))
        beta_sum = sum(val for name, val in model_res.params.items() if name.startswith("beta"))
        persistence = alpha_sum + beta_sum
        p_beta1 = float(model_res.pvalues.get("beta[1]", 1.0))

        col_left, col_right = st.columns([1.2, 1.3])
        with col_left:
            st.markdown(f"##### Paramètres GARCH({p_garch}, {q_garch}) — AIC : {model_res.aic:.1f} | BIC : {model_res.bic:.1f}")
            st.dataframe(df_params, hide_index=True, use_container_width=True)
            
            if persistence < 1.0:
                st.caption(f"✅ Processus stationnaire : $\\sum \\alpha + \\sum \\beta = {persistence:.3f} < 1$")
            else:
                st.caption(f"⚠️ Non-stationnaire : $\\sum \\alpha + \\sum \\beta = {persistence:.3f} \\ge 1$")

        with col_right:
            st.markdown("##### 💡 Bilan biomécanique")
            if p_beta1 > 0.05:
                st.info(
                    rf"""
                    * **Sensibilité immédiate ($\sum \alpha = {alpha_sum:.2f}$) :** Forte réactivité au contact.
                    * **Persistance faible ($\sum \beta = {beta_sum:.2f}$, non significatif à $p={p_beta1:.2f}$) :**  
                      La composante mémoire n'est pas significative. **Le choc physique s'apparente à un ARCH({q_garch}) pur.**
                    """
                )
            else:
                st.info(
                    rf"""
                    * **Lutte prolongée ($\sum \beta = {beta_sum:.2f}$, $p={p_beta1:.2f}$) :**  
                      Le joueur ({player_pos}) continue de pousser après contact, prolongeant la dissipation d'énergie.
                    """
                )


# ==============================================================================
# 2. CAS NFL : ANALYSE DES CHOCS ARCH
# ==============================================================================
elif mode == "Cas NFL — Volatilité Réactive (ARCH)":
    st.sidebar.subheader("Ordre du modèle ARCH")
    q_arch = st.sidebar.slider("Ordre ARCH (q) — nombre de chocs récents", min_value=1, max_value=4, value=1)
    compare_garch = st.sidebar.checkbox("Superposer la courbe GARCH(1,1)", value=True)

    st.subheader(f"Analyse d'Impact Réel : Modèle ARCH({q_arch})")

    with st.spinner("Chargement et indexation des données NFL..."):
        df_raw, df_weight, plays = load_cached_nfl_data()

    if df_raw is None or not plays:
        st.error("Données introuvables. Vérifiez la présence des fichiers dans `data/`.")
    else:
        df_plays = pd.DataFrame(plays, columns=["game_id", "play_id", "nfl_id"])
        df_plays = df_plays.merge(
            df_weight[["nfl_id", "display_name", "player_position", "mass_kg"]],
            on="nfl_id",
            how="left"
        ).fillna({"display_name": "Joueur", "player_position": "N/A"})

        df_plays["label"] = (
            df_plays["display_name"] + " (" + df_plays["player_position"] + ") | Game " + 
            df_plays["game_id"].astype(str) + " | Play " + df_plays["play_id"].astype(str)
        )

        st.sidebar.subheader("Sélection de l'action")
        selected_label = st.sidebar.selectbox("Action avec contact complet :", df_plays["label"].tolist(), key="arch_play_select")
        selected_row = df_plays[df_plays["label"] == selected_label].iloc[0]

        g_id = int(selected_row["game_id"])
        p_id = int(selected_row["play_id"])
        player_id = int(selected_row["nfl_id"])
        player_name = str(selected_row["display_name"])
        player_pos = str(selected_row["player_position"])
        mass_kg = float(selected_row["mass_kg"])

        with st.spinner(f"Calcul des forces et ajustement ARCH({q_arch})..."):
            df_play, t_start, t_end, dt_contact = arch_module.compute_player_kinematics(
                df_raw, mass_kg, g_id, p_id, player_id
            )
            res_arch, sigma_arch = arch_module.fit_arch_impact(df_play["force_decel_N"], q=q_arch)
            
            sigma_garch = None
            if compare_garch:
                _, sigma_garch = arch_module.fit_garch_impact(df_play["force_decel_N"], p=1, q=1)

        f_max = df_play["force_decel_N"].max()
        sigma_arch_max = sigma_arch.max()

        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
        kpi1.metric("Joueur", player_name)
        kpi2.metric("Poste", player_pos)
        kpi3.metric("Masse", f"{mass_kg:.0f} kg")
        kpi4.metric("Fenêtre contact", f"{dt_contact:.2f} s")
        kpi5.metric("Force max freinage", f"{f_max:.0f} N")
        kpi6.metric(r"Pic volatilité $\sigma_t$", f"{sigma_arch_max:.0f} N")

        fig_arch = arch_module.plot_arch_curve(
            df_play,
            sigma_arch,
            t_start,
            t_end,
            dt_contact,
            q=q_arch,
            sigma_garch=sigma_garch,
            player_name=player_name,
            player_position=player_pos,
        )
        st.pyplot(fig_arch)

        col_left, col_right = st.columns([1.2, 1.3])
        with col_left:
            st.markdown(f"##### Paramètres ARCH({q_arch}) — AIC : {res_arch.aic:.1f} | BIC : {res_arch.bic:.1f}")
            df_arch_params = pd.DataFrame({
                "Paramètre": res_arch.params.index,
                "Valeur": res_arch.params.values.round(4),
                "Écart-type": res_arch.std_err.values.round(4),
                "p-value": res_arch.pvalues.values.round(4),
            })
            st.dataframe(df_arch_params, hide_index=True, use_container_width=True)

        with col_right:
            st.markdown("##### 💡 Interprétation physique ARCH")
            alpha_total = sum(val for name, val in res_arch.params.items() if name.startswith("alpha"))
            st.info(
                rf"""
                * **Choc sans mémoire ($\beta = 0$) :** La volatilité ne dépend que des chocs des $q={q_arch}$ derniers dixièmes de seconde.
                * **Sensibilité globale ($\sum \alpha = {alpha_total:.2f}$) :** Dès que l'impact physique cesse, la variance retombe aussitôt au niveau du bruit de fond $\omega$.
                """
            )


# ==============================================================================
# 3. CAS NFL : MODÉLISATION DE LA VITESSE ARIMA
# ==============================================================================
elif mode == "Cas NFL — Tendance & Vitesse (ARIMA)":
    st.subheader("Modélisation ARIMA(p, d, q) sur la Vitesse de Course")
    
    with st.spinner("Chargement des données NFL..."):
        df_raw, df_weight, plays = load_cached_nfl_data()
    
    if df_raw is None or not plays:
        st.error("Données introuvables. Vérifiez la présence des fichiers dans `data/`.")
    else:
        df_plays = pd.DataFrame(plays, columns=["game_id", "play_id", "nfl_id"])
        df_plays = df_plays.merge(
            df_weight[["nfl_id", "display_name", "player_position", "mass_kg"]],
            on="nfl_id",
            how="left"
        ).fillna({"display_name": "Joueur", "player_position": "N/A"})

        df_plays["label"] = (
            df_plays["display_name"] + " (" + df_plays["player_position"] + ") | Game " + 
            df_plays["game_id"].astype(str) + " | Play " + df_plays["play_id"].astype(str)
        )
        
        st.sidebar.subheader("Sélection de l'action")
        selected_label = st.sidebar.selectbox("Action NFL :", df_plays["label"].tolist(), key="arima_play_select")
        selected_row = df_plays[df_plays["label"] == selected_label].iloc[0]

        g_id = int(selected_row["game_id"])
        p_id = int(selected_row["play_id"])
        player_id = int(selected_row["nfl_id"])
        player_name = str(selected_row["display_name"])
        player_pos = str(selected_row["player_position"])
        mass_kg = float(selected_row["mass_kg"])

        st.sidebar.subheader("Ordre du modèle ARIMA")
        p_order = st.sidebar.slider("Ordre AR (p)", min_value=0, max_value=4, value=1)
        d_order = st.sidebar.slider("Ordre I (d)", min_value=0, max_value=2, value=1)
        q_order = st.sidebar.slider("Ordre MA (q)", min_value=0, max_value=4, value=1)
        chosen_order = (p_order, d_order, q_order)

        with st.spinner(f"Ajustement ARIMA{chosen_order} sur la vitesse..."):
            df_play, t_start, t_end, dt_contact = arima.compute_player_kinematics(
                df_raw, mass_kg, g_id, p_id, player_id
            )
            arima_res, v_fitted = arima.fit_arima_speed(df_play["v_ms"], order=chosen_order)

        v_max = df_play["v_ms"].max()
        v_mean = df_play["v_ms"].mean()

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Joueur", player_name)
        k2.metric("Poste", player_pos)
        k3.metric("Masse", f"{mass_kg:.0f} kg")
        k4.metric("Fenêtre contact", f"{dt_contact:.2f} s")
        k5.metric("Vitesse max", f"{v_max:.2f} m/s")
        k6.metric("Vitesse moyenne", f"{v_mean:.2f} m/s")

        fig_arima = arima.plot_nfl_analysis(
            df_play,
            v_fitted,
            t_start,
            t_end,
            dt_contact,
            player_id,
            mass_kg,
            player_name=player_name,
            player_position=player_pos,
            order=chosen_order,
        )
        st.pyplot(fig_arima)

        col_left, col_right = st.columns([1.2, 1.3])
        with col_left:
            st.markdown(
                f"##### Paramètres ARIMA{chosen_order} — AIC : {arima_res.aic:.1f} | BIC : {arima_res.bic:.1f}"
            )
            df_arima_params = pd.DataFrame({
                "Paramètre": arima_res.param_names,
                "Coefficient": arima_res.params.values.round(4),
                "Écart-type": arima_res.bse.values.round(4),
                "p-value": arima_res.pvalues.values.round(4),
            })
            st.dataframe(df_arima_params, hide_index=True, use_container_width=True)

        with col_right:
            st.markdown("##### 💡 Interprétation cinématique")
            st.info(
                rf"""
                * **Inertie ($p={chosen_order[0]}$) :** Corrélations autorégressives de la vitesse corporelle.
                * **Ordre d'intégration ($d={chosen_order[1]}$) :** Permet de stationnariser la trajectoire de course libre.
                * **Qualité d'ajustement :** Privilégiez l'ordre $(p,d,q)$ minimisant algébriquement l'AIC ({arima_res.aic:.1f}) et le BIC ({arima_res.bic:.1f}).
                """
            )


# ==============================================================================
# 4. VALIDATION SIMULATION GARCH
# ==============================================================================
elif mode == "GARCH(1,1) — Simulation théorique":
    st.subheader("Protocole Académique de Validation par Simulation — GARCH(1,1)")
    
    st.sidebar.subheader("Hyperparamètres de simulation")
    regime = st.sidebar.selectbox("Régime théorique :", list(garch.SIMULATION_PRESETS.keys()))
    default_p = garch.SIMULATION_PRESETS[regime]
    
    c1, c2, c3 = st.sidebar.columns(3)
    omega_val = c1.number_input(r"$\omega$", value=default_p["omega"], min_value=0.001, step=0.01, format="%.3f")
    alpha_val = c2.number_input(r"$\alpha$", value=default_p["alpha"], min_value=0.01, max_value=0.5, step=0.01, format="%.2f")
    beta_val = c3.number_input(r"$\beta$", value=default_p["beta"], min_value=0.1, max_value=0.95, step=0.01, format="%.2f")
    
    t_size = st.sidebar.select_slider(
        "Taille d'échantillon $T$ :",
        options=[30, 60, 100, 200, 500, 1000, 1500, 2500],
        value=60,
        key="garch_slider_t"
    )
    seed = int(st.sidebar.number_input("Graine aléatoire", value=42, step=1, key="garch_seed"))

    with st.spinner("Simulation et estimation GARCH..."):
        res_sim = garch.evaluate_simulation(
            T=t_size,
            omega=omega_val,
            alpha=alpha_val,
            beta=beta_val,
            h_pred=5,
            random_seed=seed
        )

    true_p = res_sim["true_params"]
    est_p = res_sim["est_params"]
    bias_persistence = abs(est_p["persistence"] - true_p["persistence"])

    df_compare = pd.DataFrame({
        "Paramètre": [r"omega (\omega)", r"alpha (\alpha)", r"beta (\beta)", r"Persistance (\alpha + \beta)"],
        "Valeur Réelle": [true_p["omega"], true_p["alpha"], true_p["beta"], true_p["persistence"]],
        "Valeur Estimée (T-5)": [round(est_p["omega"], 4), round(est_p["alpha"], 4), round(est_p["beta"], 4), round(est_p["persistence"], 4)],
        "Biais Absolu": [
            round(abs(res_sim["bias"]["omega"]), 4),
            round(abs(res_sim["bias"]["alpha"]), 4),
            round(abs(res_sim["bias"]["beta"]), 4),
            round(bias_persistence, 4)
        ]
    })
    st.dataframe(df_compare, hide_index=True, use_container_width=True)

    if t_size <= 60:
        st.warning(
            rf"⚠️ **Petit échantillon ($T={t_size}$)** : Biais d'échantillonnage visible "
            rf"(|$\Delta(\alpha+\beta)$| = {bias_persistence:.3f})."
        )
    else:
        st.success(
            rf"✅ **Grand échantillon ($T={t_size}$)** : Convergence validée. "
            rf"Persistance estimée : {est_p['persistence']:.3f}."
        )

    fig_sim = garch.plot_simulation_results(res_sim)
    st.pyplot(fig_sim)

    st.subheader(r"Diagnostics d'autocorrélation ($\varepsilon_t^2$)")
    fig_acf = garch.plot_acf_pacf(res_sim["eps_full"])
    st.pyplot(fig_acf)


# ==============================================================================
# 5. VALIDATION SIMULATION ARCH
# ==============================================================================
elif mode == "ARCH(1) — Simulation théorique":
    st.subheader("Protocole Académique de Validation par Simulation — ARCH(1)")
    
    st.sidebar.subheader("Hyperparamètres de simulation ARCH")
    preset_name = st.sidebar.selectbox("Régime théorique :", list(arch_module.ARCH_SIMULATION_PRESETS.keys()))
    default_p = arch_module.ARCH_SIMULATION_PRESETS[preset_name]

    c1, c2 = st.sidebar.columns(2)
    omega_val = c1.number_input(r"$\omega$", value=default_p["omega"], min_value=0.01, step=0.05, format="%.2f")
    alpha_val = c2.number_input(r"$\alpha$", value=default_p["alpha"], min_value=0.01, max_value=0.99, step=0.05, format="%.2f")

    t_size = st.sidebar.select_slider(
        "Taille d'échantillon $T$ :",
        options=[30, 60, 100, 200, 500, 1000, 1500, 2500],
        value=60,
        key="arch_sim_slider_t"
    )
    seed = int(st.sidebar.number_input("Graine aléatoire", value=42, step=1, key="arch_sim_seed"))

    with st.spinner("Simulation et estimation ARCH..."):
        res_arch_sim = arch_module.evaluate_arch_simulation(
            T=t_size,
            omega=omega_val,
            alpha=alpha_val,
            h_pred=5,
            random_seed=seed,
        )

    true_p = res_arch_sim["true_params"]
    est_p = res_arch_sim["est_params"]

    df_compare = pd.DataFrame({
        "Paramètre": [r"omega (\omega)", r"alpha (\alpha)"],
        "Valeur Réelle": [true_p["omega"], true_p["alpha"]],
        "Valeur Estimée (T-5)": [round(est_p["omega"], 4), round(est_p["alpha"], 4)],
        "Biais Absolu": [
            round(abs(res_arch_sim["bias"]["omega"]), 4),
            round(abs(res_arch_sim["bias"]["alpha"]), 4),
        ]
    })
    st.dataframe(df_compare, hide_index=True, use_container_width=True)

    if t_size <= 60:
        st.warning(
            rf"⚠️ **Petit échantillon ($T={t_size}$)** : Biais visible sur $\alpha$ "
            rf"(|$\Delta\alpha$| = {abs(res_arch_sim['bias']['alpha']):.3f})."
        )
    else:
        st.success(
            rf"✅ **Grand échantillon ($T={t_size}$)** : Convergence validée. "
            rf"$\alpha$ estimé : {est_p['alpha']:.3f} (réel : {true_p['alpha']:.3f})."
        )

    fig_arch_sim = arch_module.plot_arch_simulation_results(res_arch_sim)
    st.pyplot(fig_arch_sim)


# ==============================================================================
# 6. VALIDATION SIMULATION ARIMA
# ==============================================================================
else:
    st.subheader("Protocole Académique de Validation par Simulation — ARIMA(p,d,q)")
    
    st.sidebar.subheader("Hyperparamètres de simulation ARIMA")
    preset_name = st.sidebar.selectbox("Régime théorique :", list(arima.SIMULATION_PRESETS.keys()))
    preset_params = arima.SIMULATION_PRESETS[preset_name]

    ar_vals = preset_params["ar"]
    ma_vals = preset_params["ma"]
    d_val = preset_params["d"]

    t_size_arima = st.sidebar.select_slider(
        "Taille d'échantillon $T$ :",
        options=[30, 60, 100, 200, 500, 1000, 1500, 2500],
        value=60,
        key="arima_slider_t"
    )
    seed_arima = int(st.sidebar.number_input("Graine aléatoire", value=42, step=1, key="arima_seed"))

    with st.spinner("Simulation et estimation ARIMA..."):
        res_arima_sim = arima.evaluate_simulation(
            T=t_size_arima,
            ar=ar_vals,
            ma=ma_vals,
            d=d_val,
            h_pred=5,
            random_seed=seed_arima,
        )

    col_met1, col_met2, col_met3 = st.columns(3)
    col_met1.metric("Ordre évalué", f"ARIMA{res_arima_sim['order']}")
    col_met2.metric("AIC (Train T-5)", f"{res_arima_sim['aic']:.2f}")
    col_met3.metric("BIC (Train T-5)", f"{res_arima_sim['bic']:.2f}")

    fig_sim_arima = arima.plot_simulation_results(res_arima_sim)
    st.pyplot(fig_sim_arima)

    st.subheader("Diagnostics d'autocorrélation (ACF / PACF sur la vitesse)")
    fig_acf_arima = arima.plot_acf_pacf(res_arima_sim["y_full"])
    st.pyplot(fig_acf_arima)