# Projet 7 — Analyse de sentiments via Deep Learning (Guide Markdown)

> **But** : Préparer Azure ML (CLI, workspace, compute), enregistrer les datasets, créer l’environnement d’exécution, lancer un job, puis démarrer une API locale avec FastAPI.
>
> **À adapter** : Remplacez les valeurs entre chevrons `<>` si nécessaire. Les commandes sont prévues pour **WSL/Linux**.

---

## 1) Pré‑requis & Installation d’Azure CLI

```bash
# 1.1 Installer curl et utilitaires
sudo apt-get update
sudo apt-get install -y curl ca-certificates gnupg lsb-release

# 1.2 Installer l’Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# 1.3 Vérifier et ajouter l’extension Azure ML
az version
az extension add -n ml -y
```

### Connexion & variables d’environnement (WSL)

```bash
# 1.4 Se connecter (ouvre un navigateur)
az login

# 1.5 Définir les variables (à personnaliser si besoin)
SUB="<SUBSCRIPTION_ID>"   # astuce: az account show --query id -o tsv
RG="rg-p7"
WS="ws-p7"
LOCATION="francecentral"

# 1.6 Sélectionner la souscription
az account set --subscription "$SUB"
```

---

## 2) Vérifications rapides & Workspace

```bash
# 2.1 Vérifier le compte et l’extension ML
az account show -o table
az extension show -n ml || az extension add -n ml -y

# 2.2 Vérifier l’existence du workspace
az ml workspace show -g "$RG" -w "$WS" -o table
```

---

## 3) Créer un cluster de calcul (CPU, gratuit au repos)

> **Note** : Ajustez la taille si nécessaire. Ici, `Standard_DS2_v2`.

```bash
az ml compute create -g "$RG" -w "$WS" \
  --name cpu-cluster \
  --type amlcompute \
  --min-instances 0 --max-instances 1 \
  --size Standard_DS2_v2
```

---

## 4) Enregistrer les datasets (train/val/test)

```bash
# 4.1 Vérifier la présence des fichiers
ds_root="/projet7/data"  # adapter si besoin
ls -lh "$ds_root"/{train.csv,val.csv,test.csv}

# 4.2 Créer les assets de données (v1)
az ml data create -g "$RG" -w "$WS" --name sentiment140-train --version 1 \
  --type uri_file --path "$ds_root/train.csv"

az ml data create -g "$RG" -w "$WS" --name sentiment140-val --version 1 \
  --type uri_file --path "$ds_root/val.csv"

az ml data create -g "$RG" -w "$WS" --name sentiment140-test --version 1 \
  --type uri_file --path "$ds_root/test.csv"
```

---

## 5) Créer l’environnement Azure ML

```bash
# 5.1 Créer l’asset d’environnement à partir d’un YAML
az ml environment create -g "$RG" -w "$WS" -f /projet7/azureml/env/environment.yml

# 5.2 Vérifier l’environnement (ex: p7-bert-env v1)
az ml environment show -g "$RG" -w "$WS" --name p7-bert-env --version 1 -o table
```

---

## 6) Lancer un job Azure ML

```bash
# 6.1 Se placer dans le dossier des jobs
cd /projet7/azureml/jobs

# 6.2 Créer le job à partir de job.yml et récupérer son nom
JOB=$(az ml job create -g "$RG" -w "$WS" -f job.yml --query name -o tsv)

# 6.3 Suivre les logs du job
az ml job stream -g "$RG" -w "$WS" -n "$JOB"
```

---

## 7) API locale (FastAPI) — Optionnel

```bash
# 7.1 Créer et activer un environnement Conda
conda create -n sentiment-api python=3.11 -y
conda activate sentiment-api

# 7.2 Installer les dépendances
pip install --upgrade pip
pip install fastapi "uvicorn[standard]"
pip install scikit-learn joblib numpy
pip install -U pytest httpx fastapi pydantic

# 7.3 Démarrer l’API (adapter le chemin si besoin)
cd ~/projet7  # <— changez ce chemin si nécessaire
uvicorn app.main:app --reload --port 8000
```

---

### Notes

* **Droits & rôles** : assurez-vous d’avoir les droits sur la souscription et le resource group.
* **Quotas** : si la création du compute échoue, vérifiez les quotas de la région `francecentral`.
* **Chemins** : adaptez les chemins (datasets, YAMLs) à votre arborescence locale.



# Déploiement Azure — Commande de démarrage & Git LFS

Ce mémo documente **la commande de démarrage Azure App Service** et **l’usage de Git LFS** pour versionner un modèle (ex. `model.keras`) et ses artefacts (ex. `tokenizer.json`) sans casser le déploiement.

---

## 1) Azure : définir la commande de démarrage


### Portail Azure
App Service → *Configuration* → *Paramètres généraux* → **Commande de démarrage** :
```bash
gunicorn -c gunicorn.conf.py app.main:app
```

### Azure CLI (équivalent)
```bash
az webapp config set   --resource-group <RG>   --name <APP_NAME>   --startup-file "gunicorn -c gunicorn.conf.py app.main:app"
```

---

## 2) Installer Git LFS (pour versionner les artefacts lourds)

### Ubuntu / WSL
```bash
sudo apt update
sudo apt install -y git-lfs
git lfs version   # vérif
git lfs install   # initialisation
```

> ⚠️ GitHub refuse les fichiers > 100 Mo hors LFS.

---

## 3) Suivre et inclure les artefacts du modèle

### 3.1 Traquer les fichiers avec LFS
```bash
# Tracker les fichiers lourds
git lfs track "app/artifacts/model.keras"
git add .gitattributes    # ajoute/maj le fichier généré par LFS
```

# Configuration Azure App Service (Linux)

Cette API FastAPI est déployée sur **Azure App Service**. Quelques **paramètres d’application** (variables d’environnement) doivent être configurés dans le portail Azure.

## Où configurer
**Portail Azure → App Service → Configuration → Paramètres d’application → Nouveau paramètre**  
Renseignez les paires `Nom` / `Valeur`, **Enregistrez**, puis **Redémarrez** l’application.

## Paramètres importants

| Nom (clé)                             | Valeur conseillée | Rôle |
|--------------------------------------|-------------------|------|
| `SEQ_LEN` *(recommandé)*             | `80`              | **Longueur de séquence attendue par le modèle** (en **tokens**, pas en caractères). Doit **correspondre exactement** à l’input du modèle Keras (ex. `model.input_shape[1] == 80`). |
| `MAX_LEN` *(héritage)*               | `80`              | Ancien nom utilisé par certains scripts. À éviter si `SEQ_LEN` est présent. Si vous conservez `MAX_LEN`, mettez **la même valeur que l’input du modèle**. |
| `CORS_ALLOW_ORIGINS`                 | `*` ou domaines   | Autorise les origines front qui appellent l’API. |
| `SCM_DO_BUILD_DURING_DEPLOYMENT`     | `1`               | Laisse Oryx installer les dépendances lors du déploiement GitHub Actions. |

### Paramètres optionnels (modèle volumineux)
| Nom                                   | Valeur            | Pourquoi |
|---------------------------------------|-------------------|----------|
| `WEBSITES_CONTAINER_START_TIME_LIMIT` | `600`             | Accorde plus de temps au conteneur pour démarrer si le premier chargement est long. |
| `WORKERS`                             | `1`               | Évite de charger le modèle en double dans plusieurs workers Gunicorn. |
| `TIMEOUT` / `GRACEFUL_TIMEOUT`        | `300`             | Délai Gunicorn plus large pendant l’inférence. |

> 💡 **Important : `SEQ_LEN` / `MAX_LEN` sont en *tokens*** (séquences après `tokenizer.texts_to_sequences`), **pas en caractères**.  
> Exemple : si le modèle a été entraîné avec `maxlen=80`, définir `255` provoquera une erreur du type  
> `ValueError: expected shape=(None, 80), found shape=(1, 255)`.

## Déterminer la bonne longueur
- La valeur cible est généralement celle utilisée à l’entraînement (`pad_sequences(..., maxlen=80)`).
- Elle peut aussi être lue dans le modèle : `model.input_shape[1]`.

## Vérifier
```bash
# Santé
curl -i https://<votre-app>.azurewebsites.net/health

# Prédiction (le 1er appel peut être un peu plus lent : chargement lazy)
curl -s -X POST "https://<votre-app>.azurewebsites.net/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"J’adore ce produit, c’est excellent !"}'


BASE="https://p7-sentiment-api.azurewebsites.net"

# Doit répondre: {"status":"stored"}
curl -i -X POST "$BASE/feedback" \
  -H "Content-Type: application/json" \
  -d '{"text":"Test bad prediction","predicted":"pos","correct":false,"note":"offensive content misclassified"}'
  
# Lancer les tests unitaires
pytest --cache-clear -q