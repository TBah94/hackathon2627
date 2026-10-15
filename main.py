import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# 1. Chargement ciblé des fichiers
tracking = pd.read_csv("game_tracking_2024.csv")

# 2. Trouver une action qui contient À LA FOIS 'first_contact' et 'tackle'
events_by_play = tracking.groupby(['game_id', 'play_id'])['event'].unique()
valid_plays = events_by_play[
    events_by_play.apply(lambda x: 'first_contact' in x and 'tackle' in x)
].index

selected_game, selected_play = valid_plays[0]

# Identifier le joueur plaqué (porteur de balle ayant la plus haute vitesse avant contact)
play_df = tracking[(tracking['game_id'] == selected_game) & (tracking['play_id'] == selected_play)]
tackled_player_id = play_df.groupby('nfl_id')['s'].max().idxmax()

# Isoler la série temporelle du joueur plaqué
player_df = play_df[play_df['nfl_id'] == tackled_player_id].sort_values('time').reset_index(drop=True)

# 3. Série d'accélération en m/s² (1 yd/s² = 0.9144 m/s²)
accel = (player_df['a'] * 0.9144).values
time_axis = np.arange(len(accel)) * 0.1  # Pas de 0.1 s

# 4. Modélisation ARIMA(1, 0, 1) + GARCH(1, 1)
arima_mod = ARIMA(accel, order=(1, 0, 1)).fit()
residuals = arima_mod.resid

garch_mod = arch_model(residuals, p=1, q=1, vol='Garch', dist='Normal').fit(disp='off')
volatility = garch_mod.conditional_volatility

# 5. Extraction des indices temporels des deux événements
idx_contact = player_df[player_df['event'] == 'first_contact'].index
idx_tackle = player_df[player_df['event'] == 'tackle'].index

t_contact = time_axis[idx_contact[0]] if len(idx_contact) > 0 else None
t_tackle = time_axis[idx_tackle[0]] if len(idx_tackle) > 0 else None

# 6. Visualisation conjointe
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

# Graphe 1 : Accélération brute vs ARIMA
ax1.plot(time_axis, accel, color="#1565c0", lw=2, label="Accélération observée $a_t$ (m/s²)")
ax1.plot(time_axis, arima_mod.fittedvalues, color="#ff9800", linestyle="--", label="Moyenne conditionnelle (ARIMA)")
ax1.set_ylabel("Accélération (m/s²)", fontweight="bold")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper left")

# Graphe 2 : Volatilité GARCH et zone d'impact
ax2.plot(time_axis, volatility, color="#d32f2f", lw=2.2, label=r"Volatilité $\sigma_t$ (GARCH)")
ax2.fill_between(time_axis, volatility, color="#d32f2f", alpha=0.15)
ax2.set_ylabel(r"Volatilité $\sigma_t$", fontweight="bold")
ax2.set_xlabel("Temps écoulé sur l'action (secondes)", fontweight="bold")
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper left")

# Colorer la zone de collision entre first_contact et tackle
if t_contact is not None and t_tackle is not None:
    for ax in (ax1, ax2):
        ax.axvspan(t_contact, t_tackle, color="gray", alpha=0.2, label="Zone de lutte / collision")
        ax.axvline(x=t_contact, color="darkorange", linestyle="-.", lw=1.8)
        ax.axvline(x=t_tackle, color="black", linestyle="-", lw=2)

    ax2.text(t_contact, ax2.get_ylim()[1]*0.8, " FIRST CONTACT", color="darkorange", fontweight="bold", rotation=90)
    ax2.text(t_tackle, ax2.get_ylim()[1]*0.8, " TACKLE", color="black", fontweight="bold", rotation=90)

plt.suptitle(f"Dynamique du choc cinétique : Détection de l'impact par GARCH\nGame {selected_game} - Play {selected_play}", 
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()