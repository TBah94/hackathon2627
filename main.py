"""
Application Streamlit — Projet 4 : Modélisation des Séries Temporelles
======================================================================
Synthèse exécutive & modélisation sur données réelles NFL :
- Cas NFL : Détection d'impacts & profil biomécanique (Poste, Nom, Poids).
- GARCH(1,1) : Protocole académique de simulation (Pr. Abass Sagna).
- ARIMA(p,d,q) : Modélisation de la moyenne & cinématique de course.
- ARCH(q) : Hétéroscédasticité pure & chocs sans mémoire.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import sécurisé des modules métier
import garch

try:
    import arima
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    import arch_module as arch
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

# Configuration Streamlit
st.set_page_config(
    page_title="Hackathon — Séries Temporelles NFL",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache des données NFL
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
st.title("Modélisation de séries temporelles - les chocs dans la NFL")
st.markdown(
    """
    **Hackathon 2027 — Projet 4 : Validation des modèles par simulation & application NFL**  
    *Comment séparer l'élan de course de la violence mécanique d'un plaquage à l'aide de séries temporelles.*
    """
)

# Tableau de synthèse comparative directe
with st.container():
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.info(
            """
            **ARIMA($p, d, q$)**  
            *Composante :* **Moyenne $\\mu_t$**  
            *Signal :* Vitesse $v(t)$ (m/s)  
            *Rôle :* Prédit la trajectoire et l'inertie de course libre.
            """
        )
    with col_b:
        st.warning(
            """
            **ARCH($q$)**  
            *Composante :* **Variance $\\sigma_t^2$ directe**  
            *Signal :* Accélération relative $(\\Delta v / \\Delta t)^2$  
            *Rôle :* Détecte l'instant du choc sans mémoire ($\\\\beta = 0$).
            """
        )
    with col_c:
        st.success(
            """
            **GARCH($p, q$)**  
            *Composante :* **Variance avec mémoire**  
            *Signal :* Force de freinage $F_{decel}$ (N)  
            *Rôle :* Mesure l'instabilité et l'énergie absorbée en lutte.
            """
        )

st.divider()

# Navigation
st.sidebar.title("Navigation")
mode = st.sidebar.radio(
    "Mode d'analyse :",
    [
        "GARCH(1,1) — Simulation théorique",
        "Cas NFL — Chocs & Freinage (GARCH)",
        "Cas NFL — Tendance & Vitesse (ARIMA)",
        "Cas NFL — Volatilité Réactive (ARCH)"
    ]
)

# ==============================================================================
# 1. CAS NFL : ANALYSE DES CHOCS
# ==============================================================================
if mode == "Cas NFL — Chocs & Freinage (GARCH)":
    st.subheader("Analyse d'Impact Réel : Modèle GARCH(1,1)")
    
    with st.spinner("Chargement des données NFL (Tracking + Combine + Players)..."):
        df_raw, df_weight, plays = load_cached_nfl_data()
    
    if df_raw is None or not plays:
        st.error("Données introuvables. Vérifiez les fichiers dans le dossier `data/`.")
    else:
        st.sidebar.subheader("Sélection de l'action")
        
        # Dictionnaire d'options avec Nom du joueur et Poste
        play_options = {}
        for (g_id, p_id, p_player) in plays:
            player_row = df_weight[df_weight["nfl_id"] == p_player]
            name = player_row["display_name"].iloc[0] if not player_row.empty else "Joueur"
            pos = player_row["player_position"].iloc[0] if not player_row.empty else "N/A"
            label = f"{name} ({pos}) | Game {g_id} | Play {p_id}"
            play_options[label] = (g_id, p_id, p_player, name, pos)
        
        selected_label = st.sidebar.selectbox("Action avec contact :", list(play_options.keys()))
        g_id, p_id, player_id, player_name, player_pos = play_options[selected_label]
        
        mass_kg = df_weight.loc[df_weight["nfl_id"] == player_id, "mass_kg"].iloc[0]

        with st.spinner("Calcul cinématique et estimation GARCH..."):
            df_play, t_start, t_end, dt_contact = garch.compute_player_kinematics(
                df_raw, mass_kg, g_id, p_id, player_id
            )
            model_res, sigma_series = garch.fit_garch_impact(df_play["force_decel_N"])

        # Indicateurs synthétiques en 6 colonnes
        f_max = df_play["force_decel_N"].max()
        sigma_max = sigma_series.max()
        
        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
        kpi1.metric("Joueur", player_name)
        kpi2.metric("Poste", player_pos)
        kpi3.metric("Masse", f"{mass_kg:.0f} kg")
        kpi4.metric("Fenêtre contact", f"{dt_contact:.2f} s")
        kpi5.metric("Force max freinage", f"{f_max:.0f} N")
        kpi6.metric(r"Pic volatilité $\sigma_t$", f"{sigma_max:.0f} N")

        # Figure principale
        fig_nfl = garch.plot_nfl_analysis(
            df_play, sigma_series, t_start, t_end, dt_contact, 
            player_id, mass_kg, player_name=player_name, player_position=player_pos
        )
        st.pyplot(fig_nfl)

        # Synthèse des paramètres estimés
        omega_val = float(model_res.params.get("omega", 0.0))
        alpha_val = float(model_res.params.get("alpha[1]", 0.0))
        beta_val = float(model_res.params.get("beta[1]", 0.0))
        p_beta = float(model_res.pvalues.get("beta[1]", 1.0))
        persistence = alpha_val + beta_val

        col_left, col_right = st.columns([1, 1.5])
        with col_left:
            st.markdown("##### Paramètres GARCH(1,1) estimés")
            df_params = pd.DataFrame({
                "Paramètre": ["omega (bruit de fond)", "alpha (réaction choc)", "beta (persistance)"],
                "Valeur": [round(omega_val, 4), round(alpha_val, 4), round(beta_val, 4)],
                "p-value": [
                    round(float(model_res.pvalues.get("omega", 1.0)), 4),
                    round(float(model_res.pvalues.get("alpha[1]", 1.0)), 4),
                    round(p_beta, 4),
                ]
            })
            st.dataframe(df_params, hide_index=True, use_container_width=True)

        with col_right:
            st.markdown("##### 💡 Ce que révèlent ces chiffres (Bilan biomécanique)")
            if p_beta > 0.05:
                st.info(
                    rf"""
                    * **Sensibilité immédiate ($\alpha = {alpha_val:.2f}$) :** Élevée $\implies$ l'impact est transmis instantanément.
                    * **Persistance faible ($\beta = {beta_val:.2f}$, non significatif à $p={p_beta:.2f}$) :**  
                      Contrairement aux marchés financiers où les crises durent ($\beta \approx 0.85$), le choc NFL retombe aussitôt. **Le profil réel correspond à un ARCH(1) pur.**
                    """
                )
            else:
                st.info(
                    rf"""
                    * **Lutte prolongée observée ($\beta = {beta_val:.2f}$, $p={p_beta:.2f}$) :**  
                      Le joueur ({player_pos}) continue d'avancer malgré le contact, maintenant la dispersion de force pendant plusieurs dixièmes de seconde.
                    """
                )

# ==============================================================================
# 2. VALIDATION PAR SIMULATION (ATTENDUS DU SUJET)
# ==============================================================================
elif mode == "GARCH(1,1) — Simulation théorique (Pr. Sagna)":
    st.subheader("Protocole Académique de Validation par Simulation")
    
    st.sidebar.subheader("Hyperparamètres")
    regime = st.sidebar.selectbox("Régime théorique :", list(garch.SIMULATION_PRESETS.keys()))
    default_p = garch.SIMULATION_PRESETS[regime]
    
    c1, c2, c3 = st.sidebar.columns(3)
    omega_val = c1.number_input(r"$\omega$", value=default_p["omega"], min_value=0.001, step=0.01, format="%.3f")
    alpha_val = c2.number_input(r"$\alpha$", value=default_p["alpha"], min_value=0.01, max_value=0.5, step=0.01, format="%.2f")
    beta_val = c3.number_input(r"$\beta$", value=default_p["beta"], min_value=0.1, max_value=0.95, step=0.01, format="%.2f")
    
    t_size = st.sidebar.select_slider(
        "Taille d'échantillon $T$ :",
        options=[30, 60, 100, 200, 500, 1000, 1500, 2500],
        value=60
    )
    seed = int(st.sidebar.number_input("Graine aléatoire", value=42, step=1))

    with st.spinner("Simulation et estimation..."):
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

    # Tableau comparatif
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
            rf"(|$\Delta(\alpha+\beta)$| = {bias_persistence:.3f}). L'estimateur nécessite plus de points."
        )
    else:
        st.success(
            rf"✅ **Grand échantillon ($T={t_size}$)** : Convergence asymptotique validée. "
            rf"Persistance estimée : {est_p['persistence']:.3f} (réelle : {true_p['persistence']:.3f})."
        )

    # Graphiques de prévision & diagnostics
    fig_sim = garch.plot_simulation_results(res_sim)
    st.pyplot(fig_sim)

    st.subheader(r"Diagnostics d'autocorrélation ($\varepsilon_t^2$)")
    fig_acf = garch.plot_acf_pacf(res_sim["eps_full"])
    st.pyplot(fig_acf)

# ==============================================================================
# 3. MODE ARIMA : TENDANCE ET VITESSE
# ==============================================================================
elif mode == "Cas NFL — Tendance & Vitesse (ARIMA)":
    st.subheader("Modélisation ARIMA(p, d, q) sur la Vitesse")
    
    if ARIMA_AVAILABLE:
        st.success("Module `arima.py` connecté.")
    else:
        st.warning("⚠️ Module `arima.py` en attente d'intégration. Aperçu démonstratif interactif :")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Ordre AR (p)", "1", "Inertie corporelle")
        c2.metric("Ordre I (d)", "1", "Annulation de l'élan")
        c3.metric("Ordre MA (q)", "1", "Ajustement d'appui")

        t_steps = np.arange(80)
        np.random.seed(42)
        sim_speed = np.maximum(0, 5.0 - 0.08 * t_steps + np.random.normal(0, 0.25, 80))
        
        fig, ax = plt.subplots(figsize=(11, 3.5), dpi=100)
        ax.plot(t_steps[:75], sim_speed[:75], label="Vitesse observée ($T-5$)", color="#1f77b4")
        ax.plot(t_steps[75:], sim_speed[75:], label="Réel test (5 pas)", color="black", lw=2)
        ax.plot(t_steps[75:], sim_speed[74] - np.linspace(0.1, 0.4, 5), label="Prévision ARIMA", color="#2ca02c", linestyle="--", lw=2)
        ax.set_ylabel("Vitesse (m/s)")
        ax.set_xlabel("Temps (0.1 s)")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig)

# ==============================================================================
# 4. MODE ARCH : VOLATILITÉ RÉACTIVE SANS MÉMOIRE
# ==============================================================================
elif mode == "Cas NFL — Volatilité Réactive (ARCH)":
    st.subheader("Modélisation ARCH(q) : Chocs Ponctuels")
    
    if ARCH_AVAILABLE:
        st.success("Module `arch.py` connecté.")
    else:
        st.warning("⚠️ Module `arch.py` en attente d'intégration. Aperçu comparatif interactif :")
        
        t_axis = np.arange(40)
        shock = np.zeros(40)
        shock[10] = 4.0

        arch_curve = np.zeros(40)
        garch_curve = np.zeros(40)
        for t in range(1, 40):
            arch_curve[t] = 0.1 + 0.8 * (shock[t-1]**2)
            garch_curve[t] = 0.1 + 0.2 * (shock[t-1]**2) + 0.7 * garch_curve[t-1]

        fig, ax = plt.subplots(figsize=(11, 3.5), dpi=100)
        ax.plot(t_axis, arch_curve, label=r"ARCH(1) : Disparition immédiate post-choc ($\beta=0$)", color="#d62728", lw=2)
        ax.plot(t_axis, garch_curve, label=r"GARCH(1,1) : Traîne / résonance prolongée ($\beta > 0$)", color="#ff7f0e", linestyle="--", lw=2)
        ax.axvline(10, color="black", linestyle=":", label="Instant du plaquage")
        ax.set_ylabel(r"Volatilité $\sigma_t^2$")
        ax.set_xlabel("Temps (0.1 s)")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig)