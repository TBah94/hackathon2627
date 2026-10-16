"""
Module ARCH(q) : Analyse d'impacts NFL & Validation par simulation
==================================================================
Pendant du module garch.py, mais avec un modèle ARCH pur (pas de terme beta) :

    eps_t = sigma_t * z_t,   z_t ~ N(0, 1)
    sigma_t^2 = omega + alpha_1 eps_{t-1}^2 + ... + alpha_q eps_{t-q}^2

Différence avec GARCH : la volatilité ne dépend QUE des chocs récents (q derniers pas).
Elle réagit vite à un impact et retombe dès que le choc sort de la fenêtre,
alors que le GARCH garde une mémoire longue via beta * sigma_{t-1}^2.

Le module est AUTONOME : les fonctions métier du module GARCH (chargement,
sélection des actions, cinématique) sont reprises à l'identique ci-dessous,
pour ne pas dépendre d'un import « from garch » (une bibliothèque pip porte
aussi ce nom et masque le fichier local).
L'ARCH est appliqué sur la même série : la force de décélération df["force_decel_N"].
Compatible avec Streamlit (retourne des figures Matplotlib sans plt.show()).
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from arch import arch_model
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


# =====================================================================
# FONCTIONS REPRISES DU MODULE GARCH (identiques)
# =====================================================================
def load_nfl_data(
    tracking_path: str, combine_path: str, players_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les données de tracking, Combine et Players avec nfl_position."""
    if not os.path.exists(combine_path):
        raise FileNotFoundError(f"Fichier Combine introuvable : {combine_path}")
    if not os.path.exists(tracking_path):
        raise FileNotFoundError(f"Fichier Tracking introuvable : {tracking_path}")

    df_combine = pd.read_csv(combine_path)
    weight_col = "combine_weight" if "combine_weight" in df_combine.columns else "weight"
    df_weight = df_combine[["nfl_id", weight_col]].dropna().copy()
    df_weight["mass_kg"] = df_weight[weight_col] * 0.45359237

    if players_path is None:
        players_path = os.path.join("data", "players.csv")
    if os.path.exists(players_path):
        df_players = pd.read_csv(players_path)
        name_col = "display_name" if "display_name" in df_players.columns else "displayName"
        pos_col = "nfl_position" if "nfl_position" in df_players.columns else (
            "player_position" if "player_position" in df_players.columns else "position")
        cols_extract, rename_map = ["nfl_id"], {}
        if name_col in df_players.columns:
            cols_extract.append(name_col); rename_map[name_col] = "display_name"
        if pos_col in df_players.columns:
            cols_extract.append(pos_col); rename_map[pos_col] = "player_position"
        df_players_sub = (df_players[cols_extract].drop_duplicates(subset=["nfl_id"])
                          .rename(columns=rename_map))
        df_weight = pd.merge(df_weight, df_players_sub, on="nfl_id", how="left")

    if "display_name" not in df_weight.columns:
        df_weight["display_name"] = "Inconnu"
    if "player_position" not in df_weight.columns:
        df_weight["player_position"] = "N/A"
    df_weight["display_name"] = df_weight["display_name"].fillna("Inconnu")
    df_weight["player_position"] = df_weight["player_position"].fillna("N/A")

    cols_tracking = ["game_id", "play_id", "nfl_id", "time", "x", "y", "s", "a", "event"]
    df_raw = pd.read_csv(tracking_path, usecols=cols_tracking)
    return df_raw, df_weight


def get_available_plays(df_raw: pd.DataFrame, df_weight: pd.DataFrame) -> List[Tuple[int, int, int]]:
    """Identifie les séquences comportant un 'first_contact' et un 'tackle'."""
    mask = df_raw["event"].isin(["first_contact", "tackle"])
    grouped = df_raw[mask].groupby(["game_id", "play_id", "nfl_id"])["event"].apply(list)
    valid_sequences = grouped[grouped.apply(lambda e: "first_contact" in e and "tackle" in e)]
    valid_ids = set(df_weight["nfl_id"].unique())
    return [(g, p, j) for (g, p, j) in valid_sequences.index if j in valid_ids]


def compute_player_kinematics(
    df_raw: pd.DataFrame, mass_kg: float, game_id: int, play_id: int, nfl_id: int
) -> Tuple[pd.DataFrame, float, float, float]:
    """Extrait la cinématique, convertit en unités SI et calcule la force de freinage."""
    df = df_raw[(df_raw["game_id"] == game_id) & (df_raw["play_id"] == play_id)
                & (df_raw["nfl_id"] == nfl_id)].copy()
    if df.empty:
        raise ValueError(f"Aucune trace trouvée pour {game_id}-{play_id}-{nfl_id}")

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df["t_sec"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()

    dt = 0.1
    df["v_ms"] = df["s"] * 0.9144
    dv_dt = df["v_ms"].diff().fillna(0.0) / dt
    df["decel_ms2"] = (-dv_dt).clip(lower=0.0)
    df["force_decel_N"] = mass_kg * df["decel_ms2"]

    fc_series = df.loc[df["event"] == "first_contact", "t_sec"]
    tk_series = df.loc[df["event"] == "tackle", "t_sec"]
    if fc_series.empty or tk_series.empty:
        raise ValueError("La séquence ne contient pas les deux tags 'first_contact' et 'tackle'.")
    t_start, t_end = min(fc_series.iloc[0], tk_series.iloc[0]), max(fc_series.iloc[0], tk_series.iloc[0])
    return df, t_start, t_end, t_end - t_start


def fit_garch_impact(series_force: pd.Series, p: int = 1, q: int = 1,
                     scale: float = 100.0) -> Tuple[Any, pd.Series]:
    """GARCH(p,q) identique au module GARCH (sert uniquement à la comparaison)."""
    scaled = series_force / scale
    res = arch_model(scaled - scaled.mean(), mean="Zero", vol="GARCH", p=p, q=q,
                     dist="Normal", rescale=False).fit(disp="off")
    return res, res.conditional_volatility * scale


# =====================================================================
# PRÉSETS DE CONFIGURATION (MÊME LOGIQUE QUE garch.py)
# =====================================================================
ARCH_SIMULATION_PRESETS: Dict[str, Dict[str, float]] = {
    "Régime stable": {
        "omega": 0.30,
        "alpha": 0.40,   # alpha < 1 : variance finie
    },
    "Cas limite / Forte réaction aux chocs": {
        "omega": 0.10,
        "alpha": 0.90,   # proche de 1 : chocs très amplifiés, queues épaisses
    },
}


# =====================================================================
# MODULE 1 — CŒUR MÉTIER NFL : ARCH SUR LA FORCE DE DÉCÉLÉRATION
# =====================================================================
def fit_arch_impact(
    series_force: pd.Series, q: int = 1, scale: float = 100.0
) -> Tuple[Any, pd.Series]:
    """Ajuste un modèle ARCH(q) sur la force de décélération pour détecter l'impact.

    Même préparation que fit_garch_impact : mise à l'échelle puis centrage.

    Args:
        series_force: Série temporelle de la force de freinage en Newtons.
        q: Nombre de chocs passés pris en compte (ordre ARCH).
        scale: Facteur d'échelle pour assurer la convergence de l'optimiseur.

    Returns:
        Tuple (modèle fitté, série de volatilité conditionnelle en Newtons).
    """
    scaled_series = series_force / scale
    mean_adjusted = scaled_series - scaled_series.mean()

    model = arch_model(
        mean_adjusted,
        mean="Zero",
        vol="ARCH",      # ARCH pur : seul le paramètre p (= q ici) est utilisé
        p=q,
        dist="Normal",
        rescale=False,
    )
    res = model.fit(disp="off")
    sigma_series = res.conditional_volatility * scale
    return res, sigma_series


def select_arch_order(
    series_force: pd.Series, q_max: int = 5, scale: float = 100.0
) -> Tuple[int, pd.DataFrame]:
    """Choisit l'ordre q de l'ARCH par le critère BIC (plus petit = meilleur).

    Returns:
        Tuple (meilleur q, tableau des critères AIC / BIC par ordre).
    """
    rows = []
    for q in range(1, q_max + 1):
        try:
            res, _ = fit_arch_impact(series_force, q=q, scale=scale)
            rows.append({"q": q, "AIC": res.aic, "BIC": res.bic})
        except Exception:
            continue
    table = pd.DataFrame(rows).sort_values("BIC").reset_index(drop=True)
    return int(table.loc[0, "q"]), table


def plot_arch_curve(
    df: pd.DataFrame,
    sigma_arch: pd.Series,
    t_start: float,
    t_end: float,
    dt_contact: float,
    q: int = 1,
    sigma_garch: Optional[pd.Series] = None,
    player_name: str = "",
    player_position: str = "",
) -> Figure:
    """Nouvelle courbe : volatilité ARCH(q) de la force de décélération.

    Volet haut : force de freinage (N) et fenêtre de contact.
    Volet bas  : volatilité conditionnelle ARCH (et GARCH en pointillés si fournie,
                 pour comparer les deux modèles sur la même action).
    """
    fig, (ax_f, ax_v) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, dpi=110)
    info_label = f"{player_name} ({player_position})" if player_name else "Joueur"

    # 1. Force de décélération
    ax_f.plot(df["t_sec"], df["force_decel_N"], color="#2ca02c", lw=1.6,
              label="Force de freinage $F_{decel}$ (N)")
    ax_f.axvspan(t_start, t_end, color="#ff7f0e", alpha=0.25,
                 label=f"Contact (Δt = {dt_contact:.2f} s)")
    ax_f.set_ylabel("Force (N)")
    ax_f.set_title(f"Force de décélération — {info_label}", fontweight="bold")
    ax_f.grid(True, linestyle=":", alpha=0.6)
    ax_f.legend(loc="upper right")

    # 2. Volatilité conditionnelle ARCH (+ GARCH en comparaison)
    ax_v.plot(df["t_sec"], sigma_arch, color="#9467bd", lw=2,
              label=rf"Volatilité ARCH({q}) $\sigma_t(F_{{decel}})$ (N)")
    if sigma_garch is not None:
        ax_v.plot(df["t_sec"], sigma_garch, color="#d62728", lw=1.6, ls="--",
                  label=r"Volatilité GARCH(1,1) (comparaison)")
    ax_v.axvspan(t_start, t_end, color="#ff7f0e", alpha=0.25)
    ax_v.set_ylabel(r"$\sigma_t$ décélération (N)")
    ax_v.set_xlabel("Temps écoulé (secondes)")
    ax_v.set_title(f"Instabilité de la force de choc modélisée par ARCH({q})",
                   fontweight="bold")
    ax_v.grid(True, linestyle=":", alpha=0.6)
    ax_v.legend(loc="upper left")

    plt.tight_layout()
    return fig


# =====================================================================
# MODULE 2 — VALIDATION PAR SIMULATION (MÊME PROTOCOLE QUE garch.py)
# =====================================================================
def simulate_arch(
    T: int, omega: float, alpha: float, random_seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Génère une trajectoire synthétique d'un ARCH(1) de taille T.

    Modèle :
        eps_t = sigma_t * z_t avec z_t ~ N(0, 1)
        sigma_t^2 = omega + alpha * eps_{t-1}^2

    Returns:
        Tuple (eps, sigma) de taille T.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    burn_in = 200
    n_total = T + burn_in
    z = np.random.normal(0, 1, size=n_total)
    eps = np.zeros(n_total)
    sigma2 = np.zeros(n_total)

    # Variance inconditionnelle stationnaire (si alpha < 1)
    sigma2[0] = omega / (1.0 - alpha) if alpha < 1.0 else omega / 0.01
    eps[0] = np.sqrt(sigma2[0]) * z[0]

    for t in range(1, n_total):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2
        eps[t] = np.sqrt(sigma2[t]) * z[t]

    return eps[burn_in:], np.sqrt(sigma2[burn_in:])


def evaluate_arch_simulation(
    T: int,
    omega: float,
    alpha: float,
    h_pred: int = 5,
    random_seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Protocole identique à evaluate_simulation (GARCH) :
    génération de taille T, découpage Train (T-5) / Test (5),
    ré-estimation de omega et alpha, prévision de la volatilité à h = 5.
    """
    if T <= h_pred + 10:
        raise ValueError(f"T ({T}) doit être strictement supérieur à {h_pred + 10}.")

    eps, sigma_true = simulate_arch(T, omega, alpha, random_seed=random_seed)

    train_size = T - h_pred
    eps_train = eps[:train_size]

    res = arch_model(eps_train, mean="Zero", vol="ARCH", p=1,
                     dist="Normal", rescale=False).fit(disp="off")
    omega_est = float(res.params.get("omega", np.nan))
    alpha_est = float(res.params.get("alpha[1]", np.nan))

    sigma_pred = np.sqrt(res.forecast(horizon=h_pred).variance.iloc[-1].values)

    return {
        "T": T,
        "h_pred": h_pred,
        "true_params": {"omega": omega, "alpha": alpha},
        "est_params": {"omega": omega_est, "alpha": alpha_est},
        "bias": {"omega": omega_est - omega, "alpha": alpha_est - alpha},
        "pvalues": res.pvalues.to_dict(),
        "eps_full": eps,
        "sigma_true": sigma_true,
        "eps_test": eps[train_size:],
        "sigma_true_test": sigma_true[train_size:],
        "sigma_pred": sigma_pred,
    }


def plot_arch_simulation_results(eval_results: Dict[str, Any]) -> Figure:
    """Trace la trajectoire ARCH simulée, la coupure T-5 / 5 et la prévision de volatilité."""
    eps = eval_results["eps_full"]
    sigma_true = eval_results["sigma_true"]
    sigma_pred = eval_results["sigma_pred"]
    T, h_pred = eval_results["T"], eval_results["h_pred"]
    train_size = T - h_pred
    time_train, time_test = np.arange(train_size), np.arange(train_size, T)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, dpi=110)

    ax1.plot(time_train, eps[:train_size], color="#1f77b4", lw=1.2,
             label=f"Processus simulé Train ($T-{h_pred}$)")
    ax1.plot(time_test, eps[train_size:], color="#ff7f0e", lw=1.4,
             label=f"Observé Test ({h_pred} pas)")
    ax1.axvline(train_size - 1, color="black", ls="--", alpha=0.7, label="Frontière Train / Test")
    ax1.set_ylabel(r"$\varepsilon_t$")
    ax1.set_title(f"Validation par simulation ARCH(1) — Échantillon T = {T}", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    ax2.plot(time_train, sigma_true[:train_size], color="gray", lw=1.2, alpha=0.7,
             label=r"Vrai $\sigma_t$ simulé")
    ax2.plot(time_test, sigma_true[train_size:], color="black", lw=1.8, label=r"Vrai $\sigma_t$ (Test)")
    ax2.plot(time_test, sigma_pred, color="#9467bd", lw=2.2, marker="o",
             label=rf"$\hat{{\sigma}}_t$ prédit ARCH (Horizon {h_pred})")
    ax2.axvline(train_size - 1, color="black", ls="--", alpha=0.7)
    ax2.set_ylabel(r"Volatilité $\sigma_t$")
    ax2.set_xlabel("Temps (indices $t$)")
    ax2.set_title(r"Prévision hors-échantillon de la variance conditionnelle (ARCH)",
                  fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    return fig


# =====================================================================
# BLOC D'EXÉCUTION AUTONOME (TEST CONSOLE HORS STREAMLIT)
# =====================================================================
if __name__ == "__main__":
    import os

    print("=" * 65)
    print("TEST MODULE ARCH — VALIDATION SIMULATION & CAS NFL")
    print("=" * 65)

    # 1. Simulation : petit T (une action) vs grand T
    preset = ARCH_SIMULATION_PRESETS["Régime stable"]
    for t_size in (60, 1500):
        r = evaluate_arch_simulation(T=t_size, omega=preset["omega"], alpha=preset["alpha"])
        print(f"\nT = {t_size}")
        print(pd.DataFrame({"Vrais": r["true_params"], "Estimés": r["est_params"],
                            "Biais": r["bias"]}).round(4))
        plot_arch_simulation_results(r).savefig(f"arch_simulation_T{t_size}.png")

    # 2. Cas NFL : ARCH sur la force de décélération d'une action avec plaquage
    tracking_file = os.path.join("data", "game_tracking_2025.csv")
    combine_file = os.path.join("data", "combine_results.csv")
    if os.path.exists(tracking_file) and os.path.exists(combine_file):
        df_raw, df_weight = load_nfl_data(tracking_file, combine_file)
        plays = get_available_plays(df_raw, df_weight)
        if plays:
            g_id, p_id, nfl_id = plays[0]
            row = df_weight.loc[df_weight["nfl_id"] == nfl_id].iloc[0]
            df_play, t_start, t_end, dt_contact = compute_player_kinematics(
                df_raw, row["mass_kg"], g_id, p_id, nfl_id)

            q_best, table = select_arch_order(df_play["force_decel_N"], q_max=4)
            print("\nChoix de l'ordre ARCH (BIC) :\n", table.round(2))
            res_arch, sigma_arch = fit_arch_impact(df_play["force_decel_N"], q=q_best)
            _, sigma_garch = fit_garch_impact(df_play["force_decel_N"])
            print(res_arch.summary())

            fig = plot_arch_curve(df_play, sigma_arch, t_start, t_end, dt_contact, q=q_best,
                                  sigma_garch=sigma_garch, player_name=row["display_name"],
                                  player_position=row["player_position"])
            fig.savefig("arch_force_decel.png")
            print("Courbe enregistrée : arch_force_decel.png")
    else:
        print("\nFichiers NFL non trouvés dans data/. Test de simulation théorique validé.")