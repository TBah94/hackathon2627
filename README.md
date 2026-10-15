# hackathon

Lien data : https://we.tl/t-2QXmZQ9FY5BHebBP

Lien descriptif data : https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/data

Lien colab: https://colab.research.google.com/drive/1QHNnCZMjQRPRFWEQJ4QEDUevqTFToY84?usp=sharing


 Détection d'Impacts Cinétiques par Modélisation ARIMA-GARCH (NFL Tracking)

Ce projet analyse les séries temporelles haute fréquence (10 Hz) issues des données de tracking de la NFL. L'objectif est de quantifier l'effort cinétique et de détecter automatiquement les impacts (`first_contact`, `tackle`) grâce à la modélisation conjointe de la moyenne conditionnelle (ARIMA) et de la volatilité résiduelle (GARCH).
---
A tester
lien du zip (drive) : https://drive.google.com/file/d/1rpc-OfrlM8aDFSPGdjByP35iuQzh3Zgc/view?usp=sharing

code colab à tester :
```
# 1. Installation de gdown (si ce n'est pas déjà fait)
!pip install -q gdown

import gdown

# 2. Mettez le lien de partage public de votre gros fichier Google Drive ici
url = "https://drive.google.com/file/d/1rpc-OfrlM8aDFSPGdjByP35iuQzh3Zgc/view?usp=sharing"
# 3. Nom du fichier de sortie sur Colab
output = "game_tracking_2024_2.csv"

# Téléchargement avec l'option --fuzzy qui contourne l'avertissement de sécurité des gros fichiers de Google
!gdown --fuzzy "{url}" -O "{output}"

print("Téléchargement du fichier de 1 Go terminé avec succès !")
```

---

## 📋 Prérequis

* **Python 3.9 à 3.12** recommandé.
* Environnement virtuel (`venv` ou `conda`).
* Les données de tracking (`game_tracking_2024.csv` ou similaire) placées à la racine ou dans un dossier dédié.

---

## 🚀 Installation rapide

### 1. Cloner ou récupérer le projet
```bash
git clone <url_du_repo>
cd <nom_du_dossier>
```

### 2. Créer et activer un environnement virtuel

* **Sur Linux / macOS :**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

* **Sur Windows (PowerShell) :**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```

### 3. Installer les dépendances
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏃 Exécution

Assurez-vous que le fichier de données (par exemple `game_tracking_2024.csv`) est présent dans le répertoire de travail, puis lancez le script d'analyse :

```bash
python main.py
```

Le script va :
1. Extraire la série d'accélération d'un joueur cible à 10 Hz.
2. Ajuster le modèle **ARIMA(1, 0, 1)** sur la dynamique moyenne.
3. Extraire les résidus et modéliser la volatilité conditionnelle via **GARCH(1, 1)**.
4. Générer la visualisation confrontant la courbe de volatilité $\sigma_t$ aux événements officiels (`first_contact` et `tackle`).

---

## 📦 Structure des dépendances

* `pandas` & `numpy` : manipulation vectorielle et découpage des séries temporelles.
* `matplotlib` : visualisations et graphiques couplés.
* `statsmodels` : tests statistiques (ADF, ARCH-LM) et ajustement ARIMA.
* `arch` : estimation et diagnostic des modèles ARCH / GARCH.
