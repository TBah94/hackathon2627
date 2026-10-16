"""
Module GARCH(1,1) : Analyse d'impacts NFL & Validation par simulation
====================================================================
Ce module fournit :
1. Les fonctions métier pour l'analyse cinématique et la détection
   de chocs/plaquages NFL via GARCH(1,1).
2. Les fonctions de validation académique par simulation (Pr. Abass Sagna) :
   estimation Train (T-5) / Test (5), diagnostics ACF/PACF et prévision.
Compatible avec Streamlit (retourne des figures Matplotlib sans plt.show()).
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from arch import arch_model
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# =====================================================================
# PRÉSETS DE CONFIGURATION (POUR STREAMLIT / TESTS)
# =====================================================================
SIMULATION_PRESETS: Dict[str, Dict[str, float]] = {
    "Régime stable": {
        "omega": 0.10,
        "alpha": 0.15,
        "beta": 0.70,  # alpha + beta = 0.85 < 1
    },
    "Cas limite / Forte persistance": {
        "omega": 0.02,
        "alpha": 0.10,
        "beta": 0.89,  # alpha + beta = 0.99 -> quasi-intégré
    },
}

SAMPLE_SIZE_PRESETS: Dict[str, int] = {
    "Petit T (action NFL ~6s à 10 Hz)": 60,
    "Grand T (convergence asymptotique)": 1500,
}


# =====================================================================
# MODULE 1 — CŒUR MÉTIER NFL (DÉTECTION DE CHOCS & PLAQUAGES)
# =====================================================================
def load_nfl_data(
    tracking_path: str, combine_path: str, players_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les données de tracking, Combine et Players avec nfl_position."""
    if not os.path.exists(combine_path):
        raise FileNotFoundError(f"Fichier Combine introuvable : {combine_path}")
    if not os.path.exists(tracking_path):
        raise FileNotFoundError(f"Fichier Tracking introuvable : {tracking_path}")

    # 1. Poids Combine
    df_combine = pd.read_csv(combine_path)
    weight_col = "combine_weight" if "combine_weight" in df_combine.columns else "weight"
    df_weight = df_combine[["nfl_id", weight_col]].dropna().copy()
    df_weight["mass_kg"] = df_weight[weight_col] * 0.45359237

    # 2. Enrichissement avec players.csv
    if players_path is None:
        players_path = os.path.join("data", "players.csv")

    if os.path.exists(players_path):
        df_players = pd.read_csv(players_path)
        
        # Détection du nom
        name_col = "display_name" if "display_name" in df_players.columns else "displayName"
        
        # Position : 'nfl_position'
        pos_col = "nfl_position" if "nfl_position" in df_players.columns else (
            "player_position" if "player_position" in df_players.columns else "position"
        )

        cols_extract = ["nfl_id"]
        rename_map = {}
        
        if name_col in df_players.columns:
            cols_extract.append(name_col)
            rename_map[name_col] = "display_name"
            
        if pos_col in df_players.columns:
            cols_extract.append(pos_col)
            rename_map[pos_col] = "player_position"

        df_players_sub = (
            df_players[cols_extract]
            .drop_duplicates(subset=["nfl_id"])
            .rename(columns=rename_map)
        )
        
        df_weight = pd.merge(df_weight, df_players_sub, on="nfl_id", how="left")
    
    # Sécurisation des colonnes si absentes
    if "display_name" not in df_weight.columns:
        df_weight["display_name"] = "Inconnu"
    if "player_position" not in df_weight.columns:
        df_weight["player_position"] = "N/A"

    df_weight["display_name"] = df_weight["display_name"].fillna("Inconnu")
    df_weight["player_position"] = df_weight["player_position"].fillna("N/A")

    # 3. Tracking 10 Hz
    cols_tracking = ["game_id", "play_id", "nfl_id", "time", "x", "y", "s", "a", "event"]
    df_raw = pd.read_csv(tracking_path, usecols=cols_tracking)

    return df_raw, df_weight

def get_available_plays(
    df_raw: pd.DataFrame, df_weight: pd.DataFrame
) -> List[Tuple[int, int, int]]:
    """Identifie les séquences complètes comportant un 'first_contact' et un 'tackle'.

    Args:
        df_raw: DataFrame brut du tracking.
        df_weight: DataFrame des poids joueurs contenant 'nfl_id'.

    Returns:
        Liste de triplets (game_id, play_id, nfl_id) éligibles.
    """
    contact_events = ["first_contact", "tackle"]
    mask = df_raw["event"].isin(contact_events)
    grouped = df_raw[mask].groupby(["game_id", "play_id", "nfl_id"])["event"].apply(list)
    valid_sequences = grouped[
        grouped.apply(lambda evts: "first_contact" in evts and "tackle" in evts)
    ]

    valid_ids = set(df_weight["nfl_id"].unique())
    eligible_plays = [
        (g_id, p_id, p_player)
        for (g_id, p_id, p_player) in valid_sequences.index
        if p_player in valid_ids
    ]
    return eligible_plays


def compute_player_kinematics(
    df_raw: pd.DataFrame,
    mass_kg: float,
    game_id: int,
    play_id: int,
    nfl_id: int,
) -> Tuple[pd.DataFrame, float, float, float]:
    """Extrait la cinématique, convertit en unités SI et calcule la force de freinage.

    Args:
        df_raw: DataFrame de tracking brut.
        mass_kg: Masse du joueur en kg.
        game_id: Identifiant du match.
        play_id: Identifiant de l'action.
        nfl_id: Identifiant du joueur.

    Returns:
        Tuple (df_play, t_start, t_end, dt_contact).
    """
    df = df_raw[
        (df_raw["game_id"] == game_id)
        & (df_raw["play_id"] == play_id)
        & (df_raw["nfl_id"] == nfl_id)
    ].copy()

    if df.empty:
        raise ValueError(f"Aucune trace trouvée pour {game_id}-{play_id}-{nfl_id}")

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df["t_sec"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()

    # Conversions SI (échantillonnage 10 Hz -> dt = 0.1 s)
    dt = 0.1
    df["v_ms"] = df["s"] * 0.9144
    dv_dt = df["v_ms"].diff().fillna(0.0) / dt
    df["decel_ms2"] = (-dv_dt).clip(lower=0.0)
    df["force_decel_N"] = mass_kg * df["decel_ms2"]

    # Fenêtre temporelle du contact
    fc_series = df.loc[df["event"] == "first_contact", "t_sec"]
    tk_series = df.loc[df["event"] == "tackle", "t_sec"]

    if fc_series.empty or tk_series.empty:
        raise ValueError("La séquence ne contient pas les deux tags 'first_contact' et 'tackle'.")

    t_fc = fc_series.iloc[0]
    t_tk = tk_series.iloc[0]
    t_start, t_end = min(t_fc, t_tk), max(t_fc, t_tk)
    dt_contact = t_end - t_start

    return df, t_start, t_end, dt_contact


def fit_garch_impact(
    series_force: pd.Series, p: int = 1, q: int = 1, scale: float = 100.0
) -> Tuple[Any, pd.Series]:
    """Ajuste un modèle GARCH(p,q) sur la force de décélération pour détecter l'impact.

    Args:
        series_force: Série temporelle de la force de freinage en Newtons.
        p: Ordre du terme autorégressif de variance (GARCH).
        q: Ordre du terme résiduel au carré (ARCH).
        scale: Facteur d'échelle pour assurer la convergence de l'optimiseur.

    Returns:
        Tuple (modèle fitté, série de volatilité conditionnelle en Newtons).
    """
    scaled_series = series_force / scale
    mean_adjusted = scaled_series - scaled_series.mean()

    model = arch_model(
        mean_adjusted,
        mean="Zero",
        vol="GARCH",
        p=p,
        q=q,
        dist="Normal",
        rescale=False,
    )
    res = model.fit(disp="off")
    sigma_series = res.conditional_volatility * scale
    return res, sigma_series


def plot_nfl_analysis(
    df: pd.DataFrame,
    sigma_series: pd.Series,
    t_start: float,
    t_end: float,
    dt_contact: float,
    player_id: int,
    mass_kg: float,
    player_name: str = "",
    player_position: str = "",
    order: Tuple[int, int] = (1, 1)
) -> Figure:
    """
    Génère la figure à 3 volets (Cinématique, Volatilité GARCH, Trajectoire 2D).
    Inclut le nom et la position du joueur dans le titre.
    """
    fig = plt.figure(figsize=(15, 8), dpi=110)
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], hspace=0.3, wspace=0.25)

    ax_kin = fig.add_subplot(gs[0, 0])
    ax_garch = fig.add_subplot(gs[1, 0], sharex=ax_kin)
    ax_map = fig.add_subplot(gs[:, 1])

    # Titre enrichi
    info_label = f"{player_name} ({player_position})" if player_name else f"Joueur {player_id}"

    # 1. Cinématique
    ax_kin.plot(df["t_sec"], df["v_ms"], color="#1f77b4", lw=1.8, label="Vitesse $v$ (m/s)")
    ax_kin.plot(
        df["t_sec"],
        df["force_decel_N"] / 100,
        color="#2ca02c",
        linestyle="--",
        lw=1.6,
        label="Force de freinage $F_{decel}$ (x100 N)",
    )
    ax_kin.axvspan(
        t_start,
        t_end,
        color="#ff7f0e",
        alpha=0.25,
        label=f"Contact (Δt = {dt_contact:.2f} s)",
    )
    ax_kin.set_ylabel("Vitesse (m/s) & Force (x100 N)")
    ax_kin.set_title(
        f"Cinématique de freinage — {info_label} | {mass_kg:.0f} kg",
        fontweight="bold",
    )
    ax_kin.grid(True, linestyle=":", alpha=0.6)
    ax_kin.legend(loc="upper right")

    # 2. Volatilité conditionnelle GARCH
    ax_garch.plot(
        df["t_sec"],
        sigma_series,
        color="#d62728",
        lw=2,
        label=r"Volatilité GARCH $\sigma_t(F_{decel})$ (N)",
    )
    ax_garch.axvspan(t_start, t_end, color="#ff7f0e", alpha=0.25)
    ax_garch.set_ylabel(r"$\sigma_t$ décélération (N)")
    ax_garch.set_xlabel("Temps écoulé (secondes)")
    ax_garch.set_title(
        f"Instabilité de la force de choc modélisée par GARCH({order[0]},{order[1]})",
        fontweight="bold",
    )
    ax_garch.grid(True, linestyle=":", alpha=0.6)
    ax_garch.legend(loc="upper left")

    # 3. Trajectoire 2D spatio-temporelle (LineCollection)
    points = np.array([df["x"], df["y"]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(df["t_sec"].min(), df["t_sec"].max())
    lc = LineCollection(segments, cmap="plasma", norm=norm, linewidth=2.8, alpha=0.95)
    lc.set_array(df["t_sec"].iloc[:-1])
    line = ax_map.add_collection(lc)

    cbar = plt.colorbar(line, ax=ax_map, orientation="horizontal", pad=0.08, shrink=0.85)
    cbar.set_label("Temps écoulé (s)", fontsize=9)

    mask_seg = (df["t_sec"] >= t_start) & (df["t_sec"] <= t_end)
    ax_map.plot(
        df.loc[mask_seg, "x"],
        df.loc[mask_seg, "y"],
        color="black",
        linewidth=4,
        linestyle="--",
        label="Lutte",
    )

    fc_row = df[df["event"] == "first_contact"].iloc[0]
    tk_row = df[df["event"] == "tackle"].iloc[0]
    ax_map.scatter(
        fc_row["x"],
        fc_row["y"],
        color="#ff7f0e",
        edgecolors="black",
        marker="X",
        s=110,
        label="First contact",
        zorder=6,
    )
    ax_map.scatter(
        tk_row["x"],
        tk_row["y"],
        color="#d62728",
        edgecolors="black",
        marker="P",
        s=110,
        label="Tackle",
        zorder=6,
    )

    pad_x, pad_y = 4.0, 3.0
    ax_map.set_xlim(max(0, df["x"].min() - pad_x), min(120, df["x"].max() + pad_x))
    ax_map.set_ylim(max(0, df["y"].min() - pad_y), min(53.33, df["y"].max() + pad_y))
    ax_map.set_xlabel("X (yards)")
    ax_map.set_ylabel("Y (yards)")
    ax_map.set_title("Trajectoire 2D terrain", fontweight="bold")
    ax_map.grid(True, linestyle=":", alpha=0.6)
    ax_map.legend(loc="best", fontsize=8)

    plt.tight_layout()
    return fig
# =====================================================================
# MODULE 2 — VALIDATION ACADÉMIQUE PAR SIMULATION (ATTENDUS DU PDF)
# =====================================================================
def simulate_garch(
    T: int, omega: float, alpha: float, beta: float, random_seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Génère une trajectoire synthétique d'un GARCH(1,1) de taille T.

    Modèle :
        eps_t = sigma_t * z_t avec z_t ~ N(0, 1)
        sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2

    Args:
        T: Nombre d'observations à simuler.
        omega: Constante du processus de variance (> 0).
        alpha: Coefficient de réaction au choc ARCH (>= 0).
        beta: Coefficient de persistance de la variance GARCH (>= 0).
        random_seed: Graine aléatoire pour reproductibilité.

    Returns:
        Tuple (eps, sigma) de taille T.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Période de chauffe (burn-in) pour éliminer l'effet de l'état initial
    burn_in = 200
    n_total = T + burn_in

    z = np.random.normal(0, 1, size=n_total)
    eps = np.zeros(n_total)
    sigma2 = np.zeros(n_total)

    # Variance inconditionnelle stationnaire (si alpha + beta < 1)
    uncond_var = omega / (1.0 - alpha - beta) if (alpha + beta < 1.0) else omega / 0.01
    sigma2[0] = max(uncond_var, 1e-4)
    eps[0] = np.sqrt(sigma2[0]) * z[0]

    for t in range(1, n_total):
        sigma2[t] = omega + alpha * (eps[t - 1] ** 2) + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * z[t]

    return eps[burn_in:], np.sqrt(sigma2[burn_in:])


def evaluate_simulation(
    T: int,
    omega: float,
    alpha: float,
    beta: float,
    h_pred: int = 5,
    random_seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Exécute le protocole académique complet :

    - Génération de taille T
    - Découpage Train (T-5) / Test (5)
    - Ré-estimation des paramètres
    - Prévision de la variance conditionnelle à horizon h=5.

    Args:
        T: Taille totale de l'échantillon.
        omega, alpha, beta: Vrais paramètres fixés pour la simulation.
        h_pred: Horizon de prévision sur l'échantillon test (défaut = 5).
        random_seed: Graine aléatoire.

    Returns:
        Dictionnaire de résultats quantitatifs et objets prédictifs.
    """
    if T <= h_pred + 10:
        raise ValueError(f"T ({T}) doit être strictement supérieur à {h_pred + 10}.")

    eps, sigma_true = simulate_garch(T, omega, alpha, beta, random_seed=random_seed)

    # Découpage strict T-5 (Train) et 5 (Test)
    train_size = T - h_pred
    eps_train = eps[:train_size]
    eps_test = eps[train_size:]
    sigma_true_test = sigma_true[train_size:]

    # Ajustement sur le bloc Train
    model = arch_model(
        eps_train,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="Normal",
        rescale=False,
    )
    res = model.fit(disp="off")

    omega_est = float(res.params.get("omega", np.nan))
    alpha_est = float(res.params.get("alpha[1]", np.nan))
    beta_est = float(res.params.get("beta[1]", np.nan))

    # Prévision récursive sur h_pred pas de temps
    forecasts = res.forecast(horizon=h_pred)
    sigma2_pred = forecasts.variance.iloc[-1].values
    sigma_pred = np.sqrt(sigma2_pred)

    results = {
        "T": T,
        "h_pred": h_pred,
        "true_params": {"omega": omega, "alpha": alpha, "beta": beta, "persistence": alpha + beta},
        "est_params": {
            "omega": omega_est,
            "alpha": alpha_est,
            "beta": beta_est,
            "persistence": alpha_est + beta_est,
        },
        "bias": {
            "omega": omega_est - omega,
            "alpha": alpha_est - alpha,
            "beta": beta_est - beta,
        },
        "pvalues": res.pvalues.to_dict(),
        "eps_full": eps,
        "sigma_true": sigma_true,
        "eps_test": eps_test,
        "sigma_true_test": sigma_true_test,
        "sigma_pred": sigma_pred,
    }
    return results


def plot_simulation_results(eval_results: Dict[str, Any]) -> Figure:
    """Trace la trajectoire simulée, la coupure T-5 / 5 et la prévision de variance."""
    eps = eval_results["eps_full"]
    sigma_true = eval_results["sigma_true"]
    sigma_pred = eval_results["sigma_pred"]
    T = eval_results["T"]
    h_pred = eval_results["h_pred"]
    train_size = T - h_pred

    time_train = np.arange(train_size)
    time_test = np.arange(train_size, T)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, dpi=110)

    # 1. Trajectoire du processus simulé
    ax1.plot(time_train, eps[:train_size], color="#1f77b4", lw=1.2, label=f"Processus simulé Train ($T-{h_pred}$)")
    ax1.plot(time_test, eps[train_size:], color="#ff7f0e", lw=1.4, label=f"Observé Test ({h_pred} pas)")
    ax1.axvline(train_size - 1, color="black", linestyle="--", alpha=0.7, label="Frontière Train / Test")
    ax1.set_ylabel(r"$\varepsilon_t$")
    ax1.set_title(
        f"Validation par simulation GARCH(1,1) — Échantillon T = {T}",
        fontweight="bold",
    )
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # 2. Volatilité vraie vs Volatilité prédite
    ax2.plot(time_train, sigma_true[:train_size], color="gray", lw=1.2, alpha=0.7, label=r"Vrai $\sigma_t$ simulé")
    ax2.plot(time_test, sigma_true[train_size:], color="black", lw=1.8, label=r"Vrai $\sigma_t$ (Test)")
    ax2.plot(time_test, sigma_pred, color="#d62728", lw=2.2, marker="o", label=r"$\hat{\sigma}_t$ prédit (Horizon 5)")
    ax2.axvline(train_size - 1, color="black", linestyle="--", alpha=0.7)
    ax2.set_ylabel(r"Volatilité $\sigma_t$")
    ax2.set_xlabel("Temps (indices $t$)")
    ax2.set_title(
        r"Capacité de prévision hors-échantillon de la variance conditionnelle $\sigma_t$",
        fontweight="bold",
    )
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    return fig


def plot_acf_pacf(series: np.ndarray, lags: int = 20) -> Figure:
    """Trace l'ACF et la PACF de la série au carré eps_t^2.

    Permet de valider la présence d'effets ARCH / hétéroscédasticité conditionnelle (Slide 5 du sujet).
    """
    squared_series = series**2
    lags = min(lags, len(series) // 3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), dpi=110)

    plot_acf(squared_series, lags=lags, ax=ax1, color="#1f77b4", auto_ylims=True)
    ax1.set_title(r"ACF de $\varepsilon_t^2$ (Effet ARCH)", fontweight="bold")
    ax1.set_xlabel("Décalage (lags)")
    ax1.grid(True, linestyle=":", alpha=0.6)

    plot_pacf(
        squared_series,
        lags=lags,
        ax=ax2,
        color="#d62728",
        method="ywm",
        auto_ylims=True,
    )
    ax2.set_title(r"PACF de $\varepsilon_t^2$", fontweight="bold")
    ax2.set_xlabel("Décalage (lags)")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    return fig


# =====================================================================
# BLOC D'EXÉCUTION AUTONOME (TEST CONSOLE HORS STREAMLIT)
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("TEST MODULE GARCH(1,1) — VALIDATION SIMULATION & CAS NFL")
    print("=" * 65)

    # 1. Test du protocole de simulation académique
    preset = SIMULATION_PRESETS["Régime stable"]
    t_size = SAMPLE_SIZE_PRESETS["Grand T (convergence asymptotique)"]

    print(f"\n[1/2] Lancement de la simulation synthétique (T={t_size})...")
    res_sim = evaluate_simulation(
        T=t_size,
        omega=preset["omega"],
        alpha=preset["alpha"],
        beta=preset["beta"],
        h_pred=5,
    )

    print("\nComparaison des paramètres :")
    df_cmp = pd.DataFrame(
        {
            "Vrais": res_sim["true_params"],
            "Estimés": res_sim["est_params"],
            "Biais": {**res_sim["bias"], "persistence": res_sim["est_params"]["persistence"] - res_sim["true_params"]["persistence"]},
        }
    )
    print(df_cmp.round(4))

    # 2. Test du chargement NFL (si les données sont présentes)
    tracking_file = (
        os.path.join("data", "game_tracking_2025.csv")
        if os.path.exists(os.path.join("data", "game_tracking_2025.csv"))
        else os.path.join("data", "games_tracking_2025.csv")
    )
    combine_file = os.path.join("data", "combine_results.csv")

    if os.path.exists(tracking_file) and os.path.exists(combine_file):
        print(f"\n[2/2] Chargement des données NFL réelles...")
        df_raw, df_weight = load_nfl_data(tracking_file, combine_file)
        plays = get_available_plays(df_raw, df_weight)
        print(f"Nombre de séquences de plaquages valides détectées : {len(plays)}")

        if plays:
            sample_g, sample_p, sample_player = plays[0]
            mass = df_weight.loc[df_weight["nfl_id"] == sample_player, "mass_kg"].iloc[0]
            df_play, t_start, t_end, dt_contact = compute_player_kinematics(
                df_raw, mass, sample_g, sample_p, sample_player
            )
            _, sigma_impact = fit_garch_impact(df_play["force_decel_N"])
            print(f"Ajustement réussi sur le joueur {sample_player} (Δt = {dt_contact:.2f} s).")
    else:
        print("\n[2/2] Fichiers NFL non trouvés dans data/. Test de simulation théorique validé.")

    print("\n✓ Module garch.py prêt à être importé dans main.py pour Streamlit.")