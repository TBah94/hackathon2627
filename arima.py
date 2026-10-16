"""
Module ARIMA(p,d,q) : Modélisation de la Vitesse NFL & Validation par simulation
================================================================================
Ce module fournit :
1. Les fonctions métier pour l'analyse cinématique et la prévision de la
   vitesse des joueurs NFL via ARIMA(p,d,q).
2. Les fonctions de validation académique par simulation :
   estimation Train (T-5) / Test (5), diagnostics ACF/PACF et prévision.
Compatible avec Streamlit (retourne des figures Matplotlib sans plt.show()).
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess


# =====================================================================
# PRÉSETS DE CONFIGURATION (POUR STREAMLIT / TESTS)
# =====================================================================
SIMULATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "Régime AR(1) stable": {
        "ar": [0.7],
        "ma": [],
        "d": 0,
    },
    "Régime ARMA(1,1) persistant": {
        "ar": [0.8],
        "ma": [-0.4],
        "d": 0,
    },
    "Régime ARIMA(1,1,1) non-stationnaire": {
        "ar": [0.5],
        "ma": [0.3],
        "d": 1,
    },
}

SAMPLE_SIZE_PRESETS: Dict[str, int] = {
    "Petit T (action NFL ~6s à 10 Hz)": 60,
    "Grand T (convergence asymptotique)": 1500,
}


# =====================================================================
# MODULE 1 — CŒUR MÉTIER NFL (MODÉLISATION DE LA VITESSE)
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

        name_col = "display_name" if "display_name" in df_players.columns else "displayName"
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
    """Identifie les séquences complètes comportant un 'first_contact' et un 'tackle'."""
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
    """Extrait la cinématique, convertit la vitesse en m/s et identifie la fenêtre de contact."""
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

    # Conversion vitesse en m/s (s est en yards/s)
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


def fit_arima_speed(
    series_vitesse: pd.Series, order: Tuple[int, int, int] = (1, 1, 1)
) -> Tuple[Any, pd.Series]:
    """Ajuste un modèle ARIMA(p,d,q) sur la vitesse du joueur.

    Args:
        series_vitesse: Série temporelle de la vitesse en m/s.
        order: Triplet (p, d, q) pour le modèle ARIMA.

    Returns:
        Tuple (modèle fitté, valeurs ajustées / prédictions in-sample).
    """
    model = ARIMA(series_vitesse, order=order)
    res = model.fit()
    fitted_values = res.fittedvalues
    return res, fitted_values


def plot_nfl_analysis(
    df: pd.DataFrame,
    v_fitted_series: pd.Series,
    t_start: float,
    t_end: float,
    dt_contact: float,
    player_id: int,
    mass_kg: float,
    player_name: str = "",
    player_position: str = "",
    order: Tuple[int, int, int] = (1, 1, 1),
) -> Figure:
    """Génère la figure à 3 volets (Cinématique Vitesse, Ajustement ARIMA, Trajectoire 2D)."""
    fig = plt.figure(figsize=(15, 8), dpi=110)
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], hspace=0.3, wspace=0.25)

    ax_kin = fig.add_subplot(gs[0, 0])
    ax_arima = fig.add_subplot(gs[1, 0], sharex=ax_kin)
    ax_map = fig.add_subplot(gs[:, 1])

    info_label = f"{player_name} ({player_position})" if player_name else f"Joueur {player_id}"

    # 1. Cinématique globale
    ax_kin.plot(df["t_sec"], df["v_ms"], color="#1f77b4", lw=1.8, label="Vitesse observée $v$ (m/s)")
    ax_kin.plot(
        df["t_sec"],
        df["decel_ms2"],
        color="#2ca02c",
        linestyle="--",
        lw=1.6,
        label="Décélération (m/s²)",
    )
    ax_kin.axvspan(
        t_start,
        t_end,
        color="#ff7f0e",
        alpha=0.25,
        label=f"Contact (Δt = {dt_contact:.2f} s)",
    )
    ax_kin.set_ylabel("Vitesse (m/s) & Decel (m/s²)")
    ax_kin.set_title(
        f"Cinématique de course — {info_label} | {mass_kg:.0f} kg",
        fontweight="bold",
    )
    ax_kin.grid(True, linestyle=":", alpha=0.6)
    ax_kin.legend(loc="upper right")

    # 2. Modélisation ARIMA de la Vitesse
    ax_arima.plot(
        df["t_sec"],
        df["v_ms"],
        color="#1f77b4",
        lw=1.5,
        alpha=0.7,
        label="Vitesse Réelle $v_t$ (m/s)",
    )
    ax_arima.plot(
        df["t_sec"],
        v_fitted_series,
        color="#d62728",
        lw=2,
        linestyle="--",
        label=f"Ajustement ARIMA{order}",
    )
    ax_arima.axvspan(t_start, t_end, color="#ff7f0e", alpha=0.25)
    ax_arima.set_ylabel("Vitesse (m/s)")
    ax_arima.set_xlabel("Temps écoulé (secondes)")
    ax_arima.set_title(
        f"Modélisation de la dynamique de vitesse par ARIMA{order}",
        fontweight="bold",
    )
    ax_arima.grid(True, linestyle=":", alpha=0.6)
    ax_arima.legend(loc="upper left")

    # 3. Trajectoire 2D spatio-temporelle
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
# MODULE 2 — VALIDATION ACADÉMIQUE PAR SIMULATION
# =====================================================================
def simulate_arima(
    T: int,
    ar: List[float],
    ma: List[float],
    d: int = 0,
    random_seed: Optional[int] = 42,
) -> np.ndarray:
    """Génère une trajectoire synthétique ARIMA(p,d,q) de taille T."""
    if random_seed is not None:
        np.random.seed(random_seed)

    # Conventions statsmodels pour ArmaProcess: ar = [1, -phi_1, -phi_2, ...], ma = [1, theta_1, ...]
    ar_coefs = np.r_[1, -np.array(ar)]
    ma_coefs = np.r_[1, np.array(ma)]

    arma_process = ArmaProcess(ar_coefs, ma_coefs)

    burn_in = 200
    series_arma = arma_process.generate_sample(nsample=T + burn_in)[burn_in:]

    if d > 0:
        # Intégration successive (cum-sum) pour obtenir une série I(d)
        series_arima = series_arma.copy()
        for _ in range(d):
            series_arima = np.cumsum(series_arima)
        return series_arima
    
    return series_arma


def evaluate_simulation(
    T: int,
    ar: List[float],
    ma: List[float],
    d: int = 0,
    h_pred: int = 5,
    random_seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Exécute le protocole académique complet pour ARIMA :

    - Génération ARIMA de taille T
    - Découpage Train (T-5) / Test (5)
    - Ré-estimation des paramètres (p, d, q)
    - Prévision de la vitesse sur horizon h=5.
    """
    if T <= h_pred + 10:
        raise ValueError(f"T ({T}) doit être strictement supérieur à {h_pred + 10}.")

    p = len(ar)
    q = len(ma)
    order = (p, d, q)

    y_full = simulate_arima(T, ar=ar, ma=ma, d=d, random_seed=random_seed)

    # Découpage strict T-5 (Train) et 5 (Test)
    train_size = T - h_pred
    y_train = y_full[:train_size]
    y_test = y_full[train_size:]

    # Ajustement sur le bloc Train
    model = ARIMA(y_train, order=order)
    res = model.fit()

    # Prévision hors-échantillon à horizon h_pred
    forecast_res = res.get_forecast(steps=h_pred)
    y_pred = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.05)

    results = {
        "T": T,
        "h_pred": h_pred,
        "order": order,
        "true_params": {"ar": ar, "ma": ma, "d": d},
        "pvalues": res.pvalues.to_dict(),
        "y_full": y_full,
        "y_test": y_test,
        "y_pred": np.asarray(y_pred),
        "conf_int": conf_int,
        "aic": res.aic,
        "bic": res.bic,
    }
    return results


def plot_simulation_results(eval_results: Dict[str, Any]) -> Figure:
    """Trace la trajectoire simulée, la coupure Train/Test et la prévision ARIMA."""
    y_full = eval_results["y_full"]
    y_pred = eval_results["y_pred"]
    conf_int = eval_results["conf_int"]
    T = eval_results["T"]
    h_pred = eval_results["h_pred"]
    order = eval_results["order"]
    train_size = T - h_pred

    time_train = np.arange(train_size)
    time_test = np.arange(train_size, T)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, dpi=110)

    # 1. Trajectoire du processus simulé
    ax1.plot(time_train, y_full[:train_size], color="#1f77b4", lw=1.2, label=f"Vitesse simulée Train ($T-{h_pred}$)")
    ax1.plot(time_test, y_full[train_size:], color="#ff7f0e", lw=1.4, label=f"Vitesse réelle Test ({h_pred} pas)")
    ax1.axvline(train_size - 1, color="black", linestyle="--", alpha=0.7, label="Frontière Train / Test")
    ax1.set_ylabel("Vitesse $v_t$")
    ax1.set_title(
        f"Validation par simulation ARIMA{order} — Échantillon T = {T}",
        fontweight="bold",
    )
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # 2. Zoom Prévision Vitesse (Test)
    ax2.plot(time_train[-20:], y_full[train_size - 20:train_size], color="#1f77b4", lw=1.2, label="Historique Train (fin)")
    ax2.plot(time_test, y_full[train_size:], color="black", lw=1.8, label="Vrai $v_t$ (Test)")
    ax2.plot(time_test, y_pred, color="#d62728", lw=2.2, marker="o", label=f"$\hat{{v}}_t$ prévision ARIMA{order}")

    # Intervalle de confiance 95%
    if conf_int is not None:
        ax2.fill_between(
            time_test,
            conf_int[:, 0],
            conf_int[:, 1],
            color="#d62728",
            alpha=0.2,
            label="IC 95%",
        )

    ax2.axvline(train_size - 1, color="black", linestyle="--", alpha=0.7)
    ax2.set_ylabel("Vitesse $v_t$")
    ax2.set_xlabel("Temps (indices $t$)")
    ax2.set_title(
        "Capacité de prévision hors-échantillon de la vitesse $v_t$",
        fontweight="bold",
    )
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    return fig


def plot_acf_pacf(series: np.ndarray, lags: int = 20) -> Figure:
    """Trace l'ACF et la PACF de la série de vitesse (ou de ses différences)."""
    lags = min(lags, len(series) // 3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), dpi=110)

    plot_acf(series, lags=lags, ax=ax1, color="#1f77b4", auto_ylims=True)
    ax1.set_title("ACF de la Vitesse", fontweight="bold")
    ax1.set_xlabel("Décalage (lags)")
    ax1.grid(True, linestyle=":", alpha=0.6)

    plot_pacf(
        series,
        lags=lags,
        ax=ax2,
        color="#d62728",
        method="ywm",
        auto_ylims=True,
    )
    ax2.set_title("PACF de la Vitesse", fontweight="bold")
    ax2.set_xlabel("Décalage (lags)")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    return fig


# =====================================================================
# BLOC D'EXÉCUTION AUTONOME (TEST CONSOLE HORS STREAMLIT)
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("TEST MODULE ARIMA(p,d,q) — VALIDATION SIMULATION & CAS NFL")
    print("=" * 65)

    # 1. Test du protocole de simulation académique
    preset = SIMULATION_PRESETS["Régime ARMA(1,1) persistant"]
    t_size = SAMPLE_SIZE_PRESETS["Grand T (convergence asymptotique)"]

    print(f"\n[1/2] Lancement de la simulation synthétique ARIMA (T={t_size})...")
    res_sim = evaluate_simulation(
        T=t_size,
        ar=preset["ar"],
        ma=preset["ma"],
        d=preset["d"],
        h_pred=5,
    )

    print(f"\nRésultats d'ajustement ARIMA{res_sim['order']} :")
    print(f"AIC : {res_sim['aic']:.2f} | BIC : {res_sim['bic']:.2f}")

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
            _, v_fitted = fit_arima_speed(df_play["v_ms"], order=(1, 1, 1))
            print(f"Ajustement ARIMA réussi sur la vitesse du joueur {sample_player} (Δt = {dt_contact:.2f} s).")
    else:
        print("\n[2/2] Fichiers NFL non trouvés dans data/. Test de simulation théorique validé.")

    print("\n✓ Module arima.py prêt à être importé dans main.py pour Streamlit.")
