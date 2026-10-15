import os
import random
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from arch import arch_model

# 1. Chemins des fichiers
COMBINE_PATH = os.path.join("data", "combine_results.csv")
TRACKING_PATH = (
    os.path.join("data", "game_tracking_2025.csv")
    if os.path.exists(os.path.join("data", "game_tracking_2025.csv"))
    else os.path.join("data", "games_tracking_2025.csv")
)

# 2. Chargement du profil physique (poids)
df_combine = pd.read_csv(COMBINE_PATH)
weight_col = "combine_weight" if "combine_weight" in df_combine.columns else "weight"
df_weight = df_combine[["nfl_id", weight_col]].dropna()

# 3. Chargement du tracking optique 10 Hz
cols_tracking = ["game_id", "play_id", "nfl_id", "time", "x", "y", "s", "a", "event"]
df_raw = pd.read_csv(TRACKING_PATH, usecols=cols_tracking)

# Recherche des séquences contenant à la fois 'first_contact' et 'tackle'
contact_events = ["first_contact", "tackle"]
contact_mask = df_raw["event"].isin(contact_events)
grouped = df_raw[contact_mask].groupby(["game_id", "play_id", "nfl_id"])["event"].apply(list)
valid_sequences = grouped[grouped.apply(lambda evts: "first_contact" in evts and "tackle" in evts)]

valid_candidates = [
    (g_id, p_id, p_player)
    for (g_id, p_id, p_player) in valid_sequences.index
    if p_player in df_weight["nfl_id"].values
]

if not valid_candidates:
    raise ValueError("Aucun candidat valide trouvé.")

# Sélection aléatoire parmi les actions éligibles
g_id, p_id, player_id = random.choice(valid_candidates)
player_weight_lbs = df_weight.loc[df_weight["nfl_id"] == player_id, weight_col].iloc[0]
mass_kg = player_weight_lbs * 0.45359237

print(f"Joueur retenu : {player_id} | {mass_kg:.1f} kg ({player_weight_lbs:.0f} lbs)")
print(f"Game: {g_id} | Play: {p_id}")

# 4. Traitement cinématique
df = df_raw[
    (df_raw["game_id"] == g_id)
    & (df_raw["play_id"] == p_id)
    & (df_raw["nfl_id"] == player_id)
].copy()

df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)
df["t_sec"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()

# Conversion SI (vitesse en m/s à 10 Hz -> dt = 0.1 s)
dt = 0.1
df["v_ms"] = df["s"] * 0.9144

# Dérivée de la vitesse : dv/dt
dv_dt = df["v_ms"].diff().fillna(0.0) / dt

# Décélération nette positive (freinage uniquement)
df["decel_ms2"] = (-dv_dt).clip(lower=0.0)

# Force de freinage / impact en Newtons : F = m * a_freinage
df["force_decel_N"] = mass_kg * df["decel_ms2"]

# Bornes de contact
t_fc = df.loc[df["event"] == "first_contact", "t_sec"].iloc[0]
t_tk = df.loc[df["event"] == "tackle", "t_sec"].iloc[0]
t_start, t_end = min(t_fc, t_tk), max(t_fc, t_tk)
dt_contact = t_end - t_start

# 5. Modélisation GARCH(1,1) sur la force de décélération
# Mise à l'échelle (x100 N) pour faciliter la convergence de l'optimiseur
scale = 100.0
series_target = df["force_decel_N"] / scale

model = arch_model(
    series_target - series_target.mean(),
    mean="Zero",
    vol="GARCH",
    p=1,
    q=1,
    dist="Normal"
)
res = model.fit(disp="off")
df["vol_decel_N"] = res.conditional_volatility * scale

# 6. Visualisation
fig = plt.figure(figsize=(15, 8), dpi=110)
gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], hspace=0.3, wspace=0.25)

ax_kin = fig.add_subplot(gs[0, 0])
ax_garch = fig.add_subplot(gs[1, 0], sharex=ax_kin)
ax_map = fig.add_subplot(gs[:, 1])

# Panel 1 : Vitesse & Force de décélération
ax_kin.plot(df["t_sec"], df["v_ms"], color="#1f77b4", lw=1.8, label="Vitesse $v$ (m/s)")
ax_kin.plot(
    df["t_sec"],
    df["force_decel_N"] / 100,
    color="#2ca02c",
    linestyle="--",
    lw=1.6,
    label="Force de freinage $F_{decel}$ (x100 N)"
)
ax_kin.axvspan(t_start, t_end, color="#ff7f0e", alpha=0.25, label=f"Contact (Δt = {dt_contact:.2f} s)")
ax_kin.set_ylabel("Vitesse (m/s) & Force (x100 N)")
ax_kin.set_title(f"Cinématique de freinage — Joueur {player_id} ({mass_kg:.0f} kg)", fontweight="bold")
ax_kin.grid(True, linestyle=":", alpha=0.6)
ax_kin.legend(loc="upper right")

# Panel 2 : Volatilité GARCH sur la décélération
ax_garch.plot(
    df["t_sec"],
    df["vol_decel_N"],
    color="#d62728",
    lw=2,
    label=r"Volatilité GARCH $\sigma_t(F_{decel})$ (N)"
)
ax_garch.axvspan(t_start, t_end, color="#ff7f0e", alpha=0.25)
ax_garch.set_ylabel(r"$\sigma_t$ décélération (N)")
ax_garch.set_xlabel("Temps écoulé (secondes)")
ax_garch.set_title("Instabilité de la force de choc modélisée par GARCH(1,1)", fontweight="bold")
ax_garch.grid(True, linestyle=":", alpha=0.6)
ax_garch.legend(loc="upper left")

# Panel 3 : Trajectoire 2D spatio-temporelle
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
    label="Lutte"
)

fc_row = df[df["event"] == "first_contact"].iloc[0]
tk_row = df[df["event"] == "tackle"].iloc[0]
ax_map.scatter(fc_row["x"], fc_row["y"], color="#ff7f0e", edgecolors="black", marker="X", s=110, label="First contact", zorder=6)
ax_map.scatter(tk_row["x"], tk_row["y"], color="#d62728", edgecolors="black", marker="P", s=110, label="Tackle", zorder=6)

pad_x, pad_y = 4.0, 3.0
ax_map.set_xlim(max(0, df["x"].min() - pad_x), min(120, df["x"].max() + pad_x))
ax_map.set_ylim(max(0, df["y"].min() - pad_y), min(53.33, df["y"].max() + pad_y))
ax_map.set_xlabel("X (yards)")
ax_map.set_ylabel("Y (yards)")
ax_map.set_title("Trajectoire 2D", fontweight="bold")
ax_map.grid(True, linestyle=":", alpha=0.6)
ax_map.legend(loc="best", fontsize=8)

plt.tight_layout()
plt.show()