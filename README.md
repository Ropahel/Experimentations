# Experimentations

Laboratoire personnel pour apprendre le reinforcement learning (RL) et les world models
(WM) avec Gymnasium et PyTorch sur CPU. Le projet est conçu pour avancer par petites
étapes : commencer par des algorithmes RL compréhensibles, enregistrer les expériences,
puis entraîner un JEPA sur les images et actions collectées.

## Décisions du projet

- **Matériel et outils :** Windows, Anaconda/Conda, Python 3.11 et PyTorch CPU. CUDA n'est
  pas utilisé.
- **Algorithmes RL :** PPO et DQN seront programmés dans le projet. Stable-Baselines3 est
  conservé comme référence indépendante pour comparer les résultats, pas comme
  implémentation de nos algorithmes.
- **Environnements :** commencer par CartPole et LunarLander, puis intégrer Toy Text et
  CarRacing progressivement. DQN sera utilisé avec des espaces d'actions discrets ;
  CarRacing et ses actions continues viendront après le socle initial.
- **Observations RL :** au début, PPO/DQN reçoivent l'observation native de l'environnement
  (par exemple le vecteur d'état de CartPole). Le rendu RGB est collecté séparément pour
  les WM.
- **Images et actions :** enregistrer chaque image RGB et chaque action, sans
  sous-échantillonnage pour le moment. La capture nécessite des environnements compatibles
  avec `render_mode="rgb_array"` ; cette compatibilité sera vérifiée par environnement.
- **Runs :** un run entraîne un modèle et possède un dossier indépendant sous
  `data/runs/<run_id>/`. Il conserve la configuration réellement utilisée, le résumé, les
  métriques, les évaluations, les vidéos, les checkpoints et le modèle final.
- **Datasets :** un fichier HDF5 indépendant par run et environnement, séparé du dossier de
  résultats, sous `<datasets_dir>/<environment_id>/<run_id>.h5`. Le chemin racine des
  datasets est configurable ; les dossiers de résultats restent sous `data/runs/`.
- **Épisodes HDF5 :** un seul fichier HDF5 par run, avec un index permettant de retrouver
  les limites des épisodes. Le fichier contient les épisodes d'entraînement et
  d'évaluation sans les distinguer. Il n'enregistre pas les récompenses ni les flags
  `terminated`/`truncated`. Un épisode de T actions peut comporter T+1 images (image
  initiale puis image après chaque action) ; leur alignement sera défini explicitement.
- **Environnements vectorisés :** commencer avec une instance d'environnement. Après
  mesure, ajouter éventuellement `gymnasium.vector.SyncVectorEnv` avec un nombre
  d'instances configurable. Un écrivain unique ajoutera les transitions au HDF5 ; il n'est
  pas prévu de faire écrire simultanément plusieurs environnements dans le même fichier.
- **World model :** après la collecte, construire une base JEPA qui encode les images et
  prédit des représentations latentes conditionnées par les actions. Aucun décodeur
  d'images n'est prévu au départ.
- **Contrôle par le JEPA, après sa base :** explorer d'une part la planification à l'aide
  du modèle de dynamique latent, et d'autre part une policy recevant l'état latent actuel
  et l'état objectif et produisant une distribution d'actions. Cette policy pourra être
  entraînée par RL. Le raisonnement hiérarchique n'est pas dans le périmètre actuel.
- **Consultation :** commandes en terminal, fichiers de résultats lisibles, TensorBoard,
  CSV et vidéos ; pas d'interface graphique prévue.
- **Sauvegardes :** vidéos et checkpoints périodiques configurables, évaluation vidéo
  finale et reprise d'un run depuis un checkpoint compatible.

Le cahier des charges et le plan détaillé sont dans
[doc/cahier-des-charges.md](doc/cahier-des-charges.md). L'organisation des runs et des
fichiers HDF5 est précisée dans [doc/architecture.md](doc/architecture.md).

## Structure du dépôt

```text
configs/
  settings.yaml          Réglages généraux et chemins de stockage
  algorithms/            Emplacement des configurations PPO/DQN
  environments/          Emplacement des configurations propres aux environnements
data/
  runs/                  Résultats générés par run (ignorés par Git)
  datasets/              HDF5 générés, séparés par environnement (ignorés par Git)
doc/
  architecture.md        Organisation des résultats et du stockage HDF5
  cahier-des-charges.md  Fonctionnalités, critères de validation et ordre d'implémentation
src/
  gymnasium_wm/          Paquet Python du projet
    environments/        Création et adaptation des environnements Gymnasium
    runs/                Cycle de vie et fichiers de résultats d'un run
    storage/             Écriture et lecture des datasets HDF5
    rl/                  Réseaux et algorithmes PPO/DQN
    evaluation/          Évaluations, métriques et enregistrement vidéo
    world_models/        Encodeur, prédicteur et entraînement JEPA
    control/             Planification et policy conditionnée par un objectif
    utils/               Graines et utilitaires partagés
tests/
  unit/                  Tests de composants isolés
  integration/           Tests de runs et d'échanges avec Gymnasium/HDF5
```

Cette arborescence prépare les responsabilités, mais les modules d'entraînement et les
configurations d'algorithmes seront créés au fil des étapes du cahier des charges. Aucun
entraînement n'est encore lancé par la structure seule.

## Préparer l'environnement avec Anaconda

L'environnement Conda du projet s'appelle `experimentations`. Il utilise Python 3.11,
PyTorch CPU et Box2D via Conda. Pip installe le projet en mode éditable et les dépendances
de `pyproject.toml`, dont Gymnasium avec Classic Control et Toy Text, ainsi que les outils
de test. Box2D est installé uniquement par Conda : il n'est pas réinstallé par pip.

### Première installation

1. Ouvrir **Anaconda Prompt** depuis le menu Démarrer.
2. Aller dans le dossier du dépôt :

   ```bat
   cd /d "C:\Users\rphld\Desktop\Projets\Experimentations"
   ```

3. Créer l'environnement :

   ```bat
   conda env create --file environment.yml
   ```

   Conda installe les dépendances Conda, puis pip installe le projet et les dépendances
   définies dans `pyproject.toml`. Laisser la commande se terminer, y compris pendant
   `Executing transaction`.
4. Activer l'environnement :

   ```bat
   conda activate experimentations
   ```

5. Vérifier Python, PyTorch et Gymnasium :

   ```bat
   where python
   python -c "import sys, torch, gymnasium; print('Python :', sys.executable); print('PyTorch :', torch.__version__); print('CUDA disponible :', torch.cuda.is_available()); print('Gymnasium :', gymnasium.__version__)"
   ```

   Le chemin Python doit contenir `envs\experimentations` et CUDA doit afficher `False`.
6. Vérifier les tests et le style :

   ```bat
   python -m pytest
   python -m ruff check .
   ```

### Utilisations suivantes

Dans Anaconda Prompt, depuis le dépôt :

```bat
cd /d "C:\Users\rphld\Desktop\Projets\Experimentations"
conda activate experimentations
```

Dans VS Code, ouvrir le dépôt, lancer **Python: Select Interpreter** avec
`Ctrl+Shift+P`, puis sélectionner l'interpréteur qui se termine par
`anaconda3\envs\experimentations\python.exe`. Ouvrir ensuite un nouveau terminal intégré
et vérifier que l'environnement est activé.

Si une création interrompue a laissé un environnement vide, le supprimer avant de
recommencer :

```bat
conda env remove -n experimentations -y
conda env create --file environment.yml
```
