"""
Module de détection de chocs & plaquages NFL par ARCH (Harmonisé)
=================================================================
Compatible avec la structure Streamlit et les fonctions de référence du module GARCH.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from arch import arch_model
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


# =====================================================================
# MODULE 1 — CHARGEMENT ET PRÉPARATION DES DONNÉES NFL (Référence)
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


def preparer_donnees_joueur(
    df_raw: pd.DataFrame, df_weight: pd.DataFrame, game_id: int, play_id: int, nfl_id: Optional[int] = None
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Optional[float], float, str, str]:
    """
    Étape 1 & 2 : Isole l'action sélectionnée, extrait la série temporelle
    d'accélération et récupère le profil morphologique du joueur.
    """
    play_df = df_raw[(df_raw['game_id'] == game_id) & (df_raw['play_id'] == play_id)]

    if play_df.empty:
        raise ValueError(f"Aucune donnée pour Game {game_id} - Play {play_id}")

    # Si le nfl_id n'est pas spécifié, on prend le joueur ayant la vitesse maximale
    if nfl_id is None:
        nfl_id = play_df.groupby('nfl_id')['s'].max().idxmax()

    player_df = play_df[play_df['nfl_id'] == nfl_id].copy()
    if player_df.empty:
        raise ValueError(f"Aucune trace pour le joueur {nfl_id} dans l'action {game_id}-{play_id}")

    player_df["time"] = pd.to_datetime(player_df["time"])
    player_df = player_df.sort_values('time').reset_index(drop=True)
    player_df["t_sec"] = (player_df["time"] - player_df["time"].iloc[0]).dt.total_seconds()

    # Récupération des données du combine / players
    infos_combine = df_weight[df_weight['nfl_id'] == nfl_id]

    poids_lbs, mass_kg = None, 80.0
    p_name, p_pos = f"Joueur {nfl_id}", "N/A"

    if not infos_combine.empty:
        w_cols = [c for c in infos_combine.columns if "weight" in c]
        if w_cols:
            poids_lbs = float(infos_combine[w_cols[0]].values[0])
            mass_kg = poids_lbs * 0.45359237
        if "display_name" in infos_combine.columns:
            p_name = str(infos_combine["display_name"].values[0])
        if "player_position" in infos_combine.columns:
            p_pos = str(infos_combine["player_position"].values[0])

    # Conversion de l'accélération en m/s² (1 yd/s² = 0.9144 m/s²)
    accel = (player_df['a'] * 0.9144).values
    time_axis = player_df["t_sec"].values

    return player_df, accel, time_axis, poids_lbs, mass_kg, p_name, p_pos


# =====================================================================
# MODULE 2 — MODÉLISATION ARCH PUR
# =====================================================================
def executer_modele_arch(accel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Étape 3 : Applique le modèle ARCH pur (moyenne constante, p=1, q=1)
    sur la série d'accélération et retourne la volatilité et la moyenne ajustée.
    """
    arch_mod = arch_model(accel, mean='Constant', lags=0, p=1, q=1, dist='Normal').fit(disp='off')
    volatility = arch_mod.conditional_volatility
    fitted_mean = accel - arch_mod.resid

    return volatility, fitted_mean, arch_mod


# =====================================================================
# MODULE 3 — VISUALISATION (COMPATIBLE STREAMLIT)
# =====================================================================
def visualiser_dynamique_choc(
    player_df: pd.DataFrame, 
    accel: np.ndarray, 
    time_axis: np.ndarray, 
    volatility: np.ndarray, 
    fitted_mean: np.ndarray, 
    poids_joueur: Optional[float], 
    game_id: int, 
    play_id: int,
    player_name: str = "",
    player_position: str = ""
) -> Figure:
    """
    Étape 4 : Génère et retourne la Figure Matplotlib comparative sans plt.show().
    """
    # Extraction des indices temporels des événements
    idx_contact = player_df[player_df['event'] == 'first_contact'].index
    idx_tackle = player_df[player_df['event'] == 'tackle'].index

    t_contact = time_axis[idx_contact[0]] if len(idx_contact) > 0 else None
    t_tackle = time_axis[idx_tackle[0]] if len(idx_tackle) > 0 else None

    # Création de la figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, dpi=110)

    # Graphe 1 : Accélération brute vs Moyenne constante
    ax1.plot(time_axis, accel, color="#1565c0", lw=2, label="Accélération observée (m/s²)")
    ax1.plot(time_axis, fitted_mean, color="#ff9800", linestyle="--", label="Moyenne constante (ARCH)")
    ax1.set_ylabel("Accélération (m/s²)", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Graphe 2 : Volatilité ARCH pure
    ax2.plot(time_axis, volatility, color="#d32f2f", lw=2.2, label="Volatilité (ARCH)")
    ax2.fill_between(time_axis, volatility, color="#d32f2f", alpha=0.15)
    ax2.set_ylabel("Volatilité", fontweight="bold")
    ax2.set_xlabel("Temps écoulé sur l'action (secondes)", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    # Coloration de la zone de collision
    if t_contact is not None and t_tackle is not None:
        for ax in (ax1, ax2):
            ax.axvspan(t_contact, t_tackle, color="gray", alpha=0.2, label="Zone de lutte / collision")
            ax.axvline(x=t_contact, color="darkorange", linestyle="-.", lw=1.8)
            ax.axvline(x=t_tackle, color="black", linestyle="-", lw=2)

        ax2.text(t_contact, ax2.get_ylim()[1]*0.8, " FIRST CONTACT", color="darkorange", fontweight="bold", rotation=90)
        ax2.text(t_tackle, ax2.get_ylim()[1]*0.8, " TACKLE", color="black", fontweight="bold", rotation=90)

    poids_str = f" - Poids : {poids_joueur} lbs" if poids_joueur else ""
    info_label = f"{player_name} ({player_position})" if player_name else "Joueur cible"

    fig.suptitle(
        f"Dynamique du choc cinétique : Détection par ARCH pur{poids_str}\n{info_label} | Game {game_id} - Play {play_id}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    return fig


# =====================================================================
# BLOC D'EXÉCUTION AUTONOME (TEST CONSOLE HORS STREAMLIT)
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("TEST MODULE ARCH — CAS NFL (ALIGNÉ SUR LE RÉFÉRENTIEL)")
    print("=" * 65)

    tracking_file = (
        os.path.join("data", "game_tracking_2025.csv")
        if os.path.exists(os.path.join("data", "game_tracking_2025.csv"))
        else os.path.join("data", "games_tracking_2025.csv")
    )
    combine_file = os.path.join("data", "combine_results.csv")

    if os.path.exists(tracking_file) and os.path.exists(combine_file):
        print("\nChargement des données de tracking et combine...")
        tracking_df, combine_df = load_nfl_data(tracking_file, combine_file)

        # Récupération des actions valides
        valid_plays = get_available_plays(tracking_df, combine_df)
        print(f"Nombre de séquences de plaquages valides détectées : {len(valid_plays)}")

        if valid_plays:
            index_action = 0
            selected_game, selected_play, selected_player = valid_plays[index_action]
            print(f"Action sélectionnée -> Index : {index_action} | Game ID : {selected_game} | Play ID : {selected_play}")

            # 4. Préparation des données et du joueur
            player_df, accel, time_axis, poids_joueur, mass_kg, p_name, p_pos = preparer_donnees_joueur(
                tracking_df, combine_df, selected_game, selected_play, selected_player
            )

            # 5. Exécution du modèle ARCH et visualisation
            volatility, fitted_mean, _ = executer_modele_arch(accel)
            fig = visualiser_dynamique_choc(
                player_df, accel, time_axis, volatility, fitted_mean, 
                poids_joueur, selected_game, selected_play, p_name, p_pos
            )
            plt.show()
    else:
        print("\nFichiers de données introuvables dans le dossier data/.")