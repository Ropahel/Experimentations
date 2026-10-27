# Gymnasium WM Lab

Un petit laboratoire pour apprendre le reinforcement learning et les world models avec
Gymnasium, sur CPU. Le projet est volontairement simple et destiné à évoluer pas à pas.

> Le dépôt contient pour le moment le squelette du projet et son environnement Python. Il
> n'y a pas encore de code d'entraînement.

## Préparer l'environnement

Python 3.11 et Conda sont utilisés. Depuis une invite Conda ouverte à la racine du dépôt :

```powershell
conda env create -f environment.yml
conda activate gymnasium-wm-lab
```

Vérifier que PyTorch utilise le CPU :

```powershell
python -c "import torch; print(torch.__version__); print('CUDA disponible :', torch.cuda.is_available())"
```

Lancer le test de fumée et le linter :

```powershell
pytest
ruff check .
```

Pour les environnements Atari, `ale-py` est installé mais les ROMs Atari ne sont pas
fournies automatiquement. Elles devront être installées séparément en respectant leur
licence. Les dépendances Box2D sont incluses pour pouvoir essayer des environnements
comme LunarLander ; cette dépendance native est installée via Conda pour éviter sa
compilation pénible sous Windows avec pip.

## Organisation des données

- `data/runs/<run_id>/` : résultats propres à un entraînement (configuration, métriques,
  poids du modèle et vidéos).
- `data/datasets/<environment_id>/` : dataset HDF5 partagé par les runs du même
  environnement. Il est séparé des dossiers de runs et ne mélange pas, par exemple,
  les images de CartPole avec celles de LunarLander.
- Le dataset est prévu pour contenir les images et actions brutes, pas les récompenses ni
  les poids des modèles.

Ces dossiers de données sont ignorés par Git afin de ne pas versionner les gros fichiers.
Les chemins généraux sont configurés dans [configs/settings.yaml](configs/settings.yaml).
Voir [doc/architecture.md](doc/architecture.md) pour les choix de structure et les
points à décider avant le premier entraînement.

## Structure actuelle

```text
configs/              paramètres généraux
data/
  datasets/           datasets HDF5, rangés par environnement
  runs/               résultats indépendants par run
doc/                  documentation du projet
src/gymnasium_wm/     code Python du projet
tests/                tests
environment.yml       environnement Conda CPU
pyproject.toml        dépendances et configuration des outils Python
```
