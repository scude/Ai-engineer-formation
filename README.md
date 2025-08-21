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
ds_root="/home/scude/projet7/data"  # adapter si besoin
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
az ml environment create -g "$RG" -w "$WS" -f /home/scude/projet7/azureml/env/environment.yml

# 5.2 Vérifier l’environnement (ex: p7-bert-env v1)
az ml environment show -g "$RG" -w "$WS" --name p7-bert-env --version 1 -o table
```

---

## 6) Lancer un job Azure ML

```bash
# 6.1 Se placer dans le dossier des jobs
cd /home/scude/projet7/azureml/jobs

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

# 7.3 Démarrer l’API (adapter le chemin si besoin)
cd ~/projet7  # <— changez ce chemin si nécessaire
uvicorn app.main:app --reload --port 8000
```

---

### Notes

* **Droits & rôles** : assurez-vous d’avoir les droits sur la souscription et le resource group.
* **Quotas** : si la création du compute échoue, vérifiez les quotas de la région `francecentral`.
* **Chemins** : adaptez les chemins (datasets, YAMLs) à votre arborescence locale.
