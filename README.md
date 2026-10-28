# Experimentations

Un petit laboratoire pour apprendre le reinforcement learning et les world models avec
Gymnasium, sur CPU. Le projet est volontairement simple et destiné à évoluer pas à pas.

## A faire

- **vérifier la version de Anaconda**
- **Préparer l'environnement virtuel, installer les dépendances**
- **Regarder la documentation de gymnaium pour savoir à quoi s'attendre et comment utiliser les environnements.**
- vérifier que tout fonctionne(au niveau des libraries)
- Faire la liste de toutes les fonctionnalité à implementer pour avoir un laboratoire complet et fonctionnel.(Pour la v1, et indiquer dans le Readme les aditions potentielles pour la V2)
- Créer tous les dossier et fichiers et indiquer à quoi ils servent.
- Faire le cahier des charges complet et finit pour la V1
- Faire une sorte de planning pour la v1, i.e, quelles fonctionnalités sont prioritaires, quelles sont les dépendances entre elles, par quoi commencer, etc.
- Coder.

## Préparer l'environnement avec Anaconda

L'environnement Conda du projet s'appelle `experimentations`. Il utilise Python 3.11,
PyTorch CPU et Box2D via Conda. Pip installe le projet en mode éditable et les dépendances
de `pyproject.toml`, dont Gymnasium avec Classic Control et Toy Text, ainsi que les outils
de test. Box2D est installé uniquement par Conda : il n'est pas réinstallé par pip.

### Première installation

1. Ouvrir **Anaconda Prompt** depuis le menu Démarrer. Il n'est pas nécessaire d'ouvrir
   une invite PowerShell ou d'installer un second environnement avec `venv`.
2. Aller dans le dossier du dépôt :

   ```bat
   cd /d "C:\Users\rphld\Desktop\Projets\Experimentations"
   ```

3. Créer l'environnement à partir du fichier du projet :

   ```bat
   conda env create --file environment.yml
   ```

   Cette étape peut prendre quelques minutes. Elle installe Conda, puis pip installe le
   projet et ses dépendances. L'installation Conda de PyTorch peut prendre du temps sous
   Windows ; laisse la commande se terminer, même si `Executing transaction` semble lent.
4. Activer l'environnement :

   ```bat
   conda activate experimentations
   ```

   Le début de l'invite doit alors afficher `(experimentations)`.
5. Vérifier que Python vient bien de cet environnement et que PyTorch utilise le CPU :

   ```bat
   where python
   python -c "import sys, torch, gymnasium; print('Python :', sys.executable); print('PyTorch :', torch.__version__); print('CUDA disponible :', torch.cuda.is_available()); print('Gymnasium :', gymnasium.__version__)"
   ```

   Le chemin de Python doit contenir `envs\experimentations`, et CUDA doit afficher
   `False`.
6. Vérifier les tests et le style du code :

   ```bat
   python -m pytest
   python -m ruff check .
   ```

### Utilisations suivantes

À chaque nouvelle session de travail, ouvrir **Anaconda Prompt**, revenir dans le dépôt
puis activer l'environnement déjà créé :

```bat
cd /d "C:\Users\rphld\Desktop\Projets\Experimentations"
conda activate experimentations
```

Il faut refaire la création avec `conda env create` uniquement si l'environnement a été
supprimé. Pour voir les environnements disponibles :

```bat
conda env list
```

Si une création est interrompue et que l'environnement apparaît dans `conda env list` mais
que `conda list -n experimentations` n'affiche aucun paquet, supprime cette création
incomplète avant de réessayer :

```bat
conda env remove -n experimentations -y
conda env create --file environment.yml
```

### Sélectionner l'environnement dans VS Code

1. Ouvrir le dossier du dépôt dans VS Code (par exemple avec `code .` depuis le dossier).
2. Appuyer sur `Ctrl+Shift+P`, chercher **Python: Select Interpreter**, puis sélectionner
   l'interpréteur qui se termine par
   `anaconda3\envs\experimentations\python.exe`.
3. Ouvrir un nouveau terminal intégré. VS Code devrait activer l'environnement choisi ;
   vérifier que le terminal commence par `(experimentations)`. Sinon, exécuter
   `conda activate experimentations` dans le terminal.

L'environnement est indépendant du dossier courant : l'activation rend ses bibliothèques
disponibles, et le `cd` place le terminal à la racine du dépôt pour exécuter les commandes
du projet.

Atari n'est pas installé pour le moment. On pourra l'ajouter plus tard ; ses ROMs ne sont
pas fournies automatiquement et devront être obtenues séparément en respectant leur
licence.

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
