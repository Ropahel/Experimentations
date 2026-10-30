# Cahier des charges — laboratoire RL et World Models

## 1. Objectif

Construire progressivement un laboratoire pédagogique sous Windows pour :

1. entraîner des modèles de RL programmés dans ce dépôt ;
2. enregistrer les résultats de chaque run séparément ;
3. collecter les images RGB et actions dans des datasets HDF5 par run ;
4. comparer les implémentations PPO/DQN aux références Stable-Baselines3 ;
5. entraîner ensuite un modèle JEPA à partir des séquences collectées ;
6. explorer, dans une phase ultérieure, la planification et une policy conditionnée par un
   objectif latent.

La priorité est la clarté et la possibilité d'inspecter les données et le code. L'objectif
n'est pas de construire une plateforme professionnelle ou distribuée.

## 2. Périmètre et décisions confirmées

### Environnement logiciel

- Windows et Anaconda/Conda.
- Python 3.11.
- PyTorch sur CPU ; pas d'utilisation de CUDA.
- Gymnasium pour les environnements.
- `h5py` pour les datasets, TensorBoard et CSV pour les métriques, imageio pour les vidéos.
- `pytest` et Ruff pour tests et qualité de code.

### Algorithmes et environnement

- Implémenter PPO et DQN dans le projet ; ne pas déléguer leur apprentissage à SB3.
- Utiliser SB3 pour une comparaison de référence sous un protocole identique.
- Ordre d'intégration : CartPole, LunarLander, puis environnements Toy Text et CarRacing.
- Utiliser les observations natives (vecteurs pour les premiers environnements) comme entrée
  RL initiale.
- Enregistrer séparément les rendus RGB pour les futurs world models.
- Commencer avec une seule instance d'environnement. Évaluer `SyncVectorEnv` ultérieurement
  par benchmark ; ne pas complexifier les premières boucles d'entraînement.
- DQN est initialement limité aux espaces d'actions discrets. L'extension aux espaces
  continus et à CarRacing sera planifiée lorsque le socle sera validé.

### Runs et résultats

- Un run entraîne un modèle avec une configuration identifiée et figée.
- Les résultats sont dans `data/runs/<run_id>/`.
- Un dossier de run doit permettre de retrouver la configuration, les métriques, le
  résumé, les évaluations, les vidéos, les checkpoints et le modèle final.
- Les sauvegardes sont périodiques et configurables. Une reprise depuis un checkpoint
  compatible est prévue.
- Les dossiers de runs restent sous `data/`, contrairement à la racine des datasets qui
  doit être configurable.

### Datasets HDF5

- Un fichier HDF5 par run et environnement, à
  `<datasets_dir>/<environment_id>/<run_id>.h5`.
- Le fichier contient les images RGB et actions de chaque épisode ; chaque transition est
  enregistrée, sans sous-échantillonnage initial.
- Les épisodes d'entraînement et d'évaluation y sont tous inclus, sans identifiant
  permettant de les différencier.
- Pas de stockage des récompenses, des poids, de `terminated` ou de `truncated` dans le
  dataset initial.
- Un index de frontières d'épisodes est obligatoire. Il ne s'agit pas d'une étiquette de
  type d'épisode, mais de la structure nécessaire pour reconstruire les séquences.
- L'alignement retenu doit représenter le fait que T actions peuvent correspondre à T+1
  images (état visuel initial et image résultant de chaque action).
- Un seul écrivain HDF5 à la fois. En cas de vectorisation, les transitions sont
  sérialisées vers cet écrivain.

### World model et contrôle latent

- Le premier modèle visé est un JEPA, pas un autoencodeur avec reconstruction d'image.
- Base minimale : encodeur d'images, prédicteur de latents conditionné par les actions,
  cible latente et objectif d'apprentissage documentés.
- Aucun décodeur d'images n'est dans le périmètre initial.
- Les deux directions ultérieures pour atteindre un objectif sont :
  1. planifier des actions avec le modèle de dynamique latent ;
  2. apprendre par RL une policy qui reçoit le latent courant et le latent objectif et
     renvoie une distribution sur les actions.
- Pas de raisonnement hiérarchique ou de politique multi-niveaux à ce stade.

## 3. Utilisation attendue

L'interface initiale est en ligne de commande. L'utilisateur doit pouvoir :

1. choisir un fichier de configuration pour un run ;
2. lancer un run et voir une progression (par exemple avec `tqdm`) ;
3. arrêter proprement un run ou reprendre depuis un checkpoint compatible ;
4. ouvrir TensorBoard et consulter le CSV ;
5. regarder les vidéos d'évaluation ;
6. retrouver le modèle final, la configuration figée et le HDF5 produit ;
7. lancer séparément une référence SB3 avec les mêmes conditions ;
8. lancer plus tard l'entraînement JEPA sur un ou plusieurs HDF5 sélectionnés.

Les commandes exactes et leurs options seront définies pendant la première étape de
l'implémentation ; elles ne doivent pas être annoncées comme disponibles avant d'être
codées et testées.

## 4. Structure cible des données

### Dossier d'un run

```text
data/runs/<run_id>/
├── config.yaml          Configuration résolue et figée
├── summary.json         État final, durée et résumé des résultats
├── metrics.csv          Historique tabulaire des métriques
├── tensorboard/         Événements consultables dans TensorBoard
├── checkpoints/         Sauvegardes périodiques pour évaluer/reprendre
├── final_model/         Poids et métadonnées du modèle final
├── evaluations/         Résultats numériques d'évaluation
└── videos/              Vidéos périodiques d'évaluation
```

`summary.json` contient au minimum l'identifiant du run, l'environnement, l'algorithme,
les graines, le statut (en cours, terminé, interrompu ou échoué), la durée, le budget
effectivement consommé, le chemin du HDF5 et des statistiques de récompense d'évaluation.
Il n'enregistre pas de valeurs inventées en cas d'échec : une erreur doit être signalée et
le statut doit rester explicite.

Le nombre d'épisodes, le meilleur score et les moyennes doivent préciser s'ils décrivent
l'entraînement ou l'évaluation dans les fichiers de métriques/résultats. Cette distinction
ne sera pas recopiée dans les épisodes HDF5.

### Fichier HDF5

Le schéma logique doit fournir :

- un tableau/dataset d'images RGB ;
- un tableau/dataset d'actions ;
- un index des épisodes indiquant pour chacun les positions de début et longueurs
  correspondantes dans les tableaux ;
- des attributs de fichier minimaux : version de schéma, environnement, identifiant du run
  et dimensions/dtypes des données.

L'index ne contient pas de champ train/evaluation. Il n'y a pas de champ
`terminated`/`truncated`. La représentation interne exacte (tableaux plats extensibles,
compression, dimension de lot d'écriture) sera choisie après un test mesurant taille,
vitesse d'écriture et lecture sur CartPole. Les séquences lues ne doivent jamais traverser
une frontière d'épisode.

Les actions peuvent être discrètes ou continues selon l'environnement ; le schéma doit
conserver leur représentation native sans les convertir en labels textuels. Les images
seront stockées dans un format de type RGB `uint8` si les rendus Gymnasium le confirment.

## 5. Exigences fonctionnelles détaillées

### F1 — Configuration

- Charger les réglages généraux depuis `configs/settings.yaml`.
- Charger une configuration spécifique au run et appliquer une règle de priorité
  déterministe entre valeurs générales et valeurs du run.
- Valider les clés, types, plages et combinaisons avant de créer l'environnement.
- Fournir des erreurs explicites indiquant le nom de la clé invalide et la valeur attendue.
- Enregistrer dans le run la configuration complètement résolue ; une modification
  ultérieure du fichier source ne doit pas changer la configuration historique.
- La graine, le device CPU, l'environnement, l'algorithme, le budget d'entraînement, les
  fréquences d'évaluation/sauvegarde/vidéo et les réglages de collecte doivent être
  configurables.
- Le chemin des résultats est fixé sous `data/runs/`; le chemin racine des datasets est
  réglable globalement.

**Validation :** configurations valides chargées comme attendu ; clés manquantes, types
incorrects, algorithme incompatible et chemin non accessible refusés avec une erreur
explicite avant l'entraînement.

### F2 — Création d'un run

- Générer un identifiant unique stable et un nouveau dossier de résultats.
- Refuser l'écrasement d'un run existant, sauf action explicite de reprise.
- Créer le HDF5 sous le dossier de l'environnement, avec le même identifiant de run.
- Écrire la configuration résolue et un statut initial avant les interactions.
- Mettre à jour progression et statut, puis écrire un résumé final.
- Fermer l'environnement, les writers vidéo, HDF5 et TensorBoard en sortie normale,
  interruption ou erreur.
- Ne pas cacher une erreur sous un statut de réussite.

**Validation :** un run minimal crée la structure attendue, les fichiers d'identification
correspondent et les ressources sont fermées même si une erreur de collecte survient.

### F3 — Environnements Gymnasium

- Construire un environnement à partir de son ID, avec `render_mode="rgb_array"` lorsque
  la capture est activée.
- Documenter pour chaque environnement : ID, type d'espace d'observation, type d'espace
  d'action, limites/résolution de rendu, dépendances additionnelles, mode d'action DQN
  possible et particularités de reset/rendu.
- Vérifier les observations, actions et images dès le premier épisode.
- Utiliser les API Gymnasium actuelles (`reset` et `step`, y compris les résultats séparés
  `terminated` et `truncated`), même si les deux flags ne sont pas stockés en HDF5.
- Fixer les graines de l'environnement et des composants aléatoires de l'algorithme.
- Garantir la fermeture de l'environnement.

**Validation :** smoke test déterministe ou reproductible au mieux pour CartPole et
LunarLander, et rendu RGB conforme au contrat.

### F4 — Collecteur et HDF5

- Collecter une image initiale après reset, puis l'action choisie et l'image obtenue après
  chaque transition, selon un alignement documenté.
- Enregistrer toutes les images/actions du run, y compris les épisodes d'évaluation, sans
  distinguer ces derniers dans le HDF5.
- Créer/ouvrir le fichier avec une version de schéma explicite et éviter qu'une relance
  accidentelle écrase un dataset existant.
- Compléter l'index d'épisode seulement lorsque ses bornes sont connues ; laisser détectable
  toute écriture interrompue.
- Écrire par blocs pour ne pas accumuler toutes les images en RAM.
- Valider les dimensions, types, longueurs et plages de valeurs ; rapporter clairement une
  image absente ou invalide.
- Documenter que les épisodes d'évaluation sont mélangés sans étiquette aux données
  d'entraînement. Une future séparation nécessitera un changement de schéma et/ou une
  nouvelle collecte.

**Validation :** écrire plusieurs épisodes de longueurs différentes, fermer/réouvrir le
fichier, retrouver les frontières exactes et confirmer qu'aucune séquence ne relie deux
épisodes. Contrôler que les actions sont T et les images T+1 selon le contrat d'alignement.

### F5 — Boucle d'apprentissage et réseaux

- Séparer les réseaux, la logique d'algorithme, la collecte et l'orchestration du run.
- Utiliser les espaces Gymnasium pour déterminer la forme des observations et des actions.
- Fournir les fonctions permettant d'évaluer une politique déterministement ou de
  l'échantillonner depuis sa distribution.
- Respecter le device CPU et enregistrer les paramètres du réseau et de l'optimiseur dans
  les checkpoints.
- Garder les tailles de réseau/hyperparamètres configurables ; ne pas coder les valeurs de
  recherche dans les boucles d'entraînement.

#### DQN

- Limité au départ aux actions discrètes.
- Implémenter réseau Q, sélection d'actions avec exploration configurable, replay buffer,
  mise à jour par mini-lot, réseau cible et perte de Bellman.
- Exposer au minimum epsilon, taille du buffer, warm-up, fréquence de mise à jour, taux
  d'apprentissage, discount et fréquence de mise à jour du réseau cible.
- Journaliser perte et statistiques d'interaction, sans écrire les récompenses dans le
  HDF5.
- Un checkpoint de reprise doit conserver réseau en ligne/cible, optimiseur, compteur,
  état d'exploration, graines et buffer si nécessaire pour une reprise cohérente.

**Validation :** tests unitaires du buffer, de la sélection d'action et du calcul de cible ;
petit entraînement CartPole sans erreur et métriques finies.

#### PPO

- Implémenter politique et valeur, collecte de trajectoires, calcul des retours/avantages,
  objectif PPO tronqué (clipping), mises à jour minibatch et optimiseur.
- Prévoir les distributions compatibles avec les premiers espaces d'actions discrets.
- Rendre configurables le taux d'apprentissage, discount, lambda GAE, clip range, taille de
  rollout, epochs, minibatch et coefficient d'entropie.
- Exposer politique d'évaluation déterministe et politique d'échantillonnage.
- Le checkpoint contient réseaux, optimiseur, compteur, état RNG et les données de rollout
  lorsqu'elles sont nécessaires à une reprise exacte.

**Validation :** tests des formes, log-probabilités, avantages et perte ; petit entraînement
CartPole sans erreur et métriques finies.

### F6 — Métriques, évaluation, vidéo

- Évaluer selon une fréquence et un nombre d'épisodes configurables, avec graines dédiées
  reproductibles.
- Séparer les métriques d'entraînement de celles d'évaluation dans CSV/TensorBoard.
- Enregistrer au minimum : pas d'environnement, épisode, récompense cumulée, longueur,
  pertes disponibles, temps écoulé et résultats d'évaluation.
- Produire une vidéo pour certains épisodes d'évaluation, pas pour chaque épisode
  d'entraînement ; fréquence configurable et vidéo finale d'évaluation.
- Les images/actions des épisodes évalués sont tout de même ajoutées au HDF5 sans tag de
  catégorie.
- Clore proprement les fichiers TensorBoard/CSV/vidéo.

**Validation :** valeurs et pas global cohérents dans CSV/TensorBoard ; vidéo lisible ;
les épisodes train et evaluation apparaissent dans le HDF5 sans champ qui les différencie.

### F7 — Checkpoint et reprise

- Sauvegarder périodiquement selon une configuration.
- Écrire les checkpoints de façon à détecter une écriture incomplète (par exemple fichier
  temporaire puis renommage).
- Reprendre uniquement à partir d'un checkpoint compatible avec l'algorithme, l'environnement
  et les dimensions du réseau.
- En cas de mismatch, afficher une erreur précise et ne pas démarrer un run neuf présenté
  comme une reprise.
- Enregistrer les compteurs d'interactions et d'optimisation, états des réseaux et
  optimiseurs, buffers nécessaires, RNG et configuration.
- Garder les checkpoints historiques selon une politique de rétention configurable et le
  modèle final séparé.

**Validation :** interruption puis reprise d'un petit run ; compter les interactions
effectives sans remise à zéro et vérifier l'incompatibilité contrôlée d'un checkpoint.

### F8 — Référence Stable-Baselines3

- Lancer SB3 comme backend de référence séparé, avec les mêmes environnements, seeds,
  budgets, fréquence d'évaluation et nombre d'épisodes d'évaluation.
- Garder ses résultats dans un dossier de run normal et enregistrer explicitement le nom /
  la version de SB3.
- Ne pas comparer des scores sans afficher les conditions de l'expérience.

**Validation :** lancer une petite référence et vérifier qu'elle produit les mêmes fichiers
de résultats attendus, avec une configuration distinguant clairement son backend.

### F9 — Vectorisation conditionnelle

- Ne pas faire dépendre la première boucle d'entraînement de `SyncVectorEnv`.
- Ajouter la vectorisation seulement après un benchmark de la version simple sur CartPole
  et LunarLander.
- Rendre le nombre d'instances configurable ; vérifier comment Gymnasium signale les
  observations après reset automatique et conserver l'association observation/action/image
  par instance.
- Le collecteur rassemble les transitions et les passe à un seul writer HDF5 ; aucun accès
  simultané non coordonné au même fichier.
- Comparer vitesse d'interaction, vitesse de mise à jour, temps total et mémoire avec
  plusieurs graines avant de retenir un défaut vectorisé.

**Validation :** pas de mélange entre frontières d'épisodes de deux instances ; ordre et
comptes de transitions vérifiés ; benchmark reproductible documenté.

### F10 — Dataset séquentiel et JEPA

- Implémenter un lecteur HDF5 qui parcourt plusieurs fichiers choisis et fournit des
  séquences d'images/actions de longueur configurable.
- Ne jamais échantillonner une fenêtre qui franchit une frontière d'épisode.
- Vérifier environnement, forme d'image, type d'action et version de schéma à l'ouverture.
- Implémenter l'encodeur, le modèle cible, le prédicteur conditionné par les actions, la
  perte JEPA et une boucle d'entraînement CPU.
- Configurer longueur de séquence, horizon, dimension latente, tailles de batch, taux
  d'apprentissage et sauvegardes.
- Écrire les runs JEPA dans `data/runs/<run_id>/` comme les runs RL ; référencer les
  fichiers source et leur version de schéma dans la configuration.
- Journaliser la perte et sauvegarder le modèle. Visualiser des embeddings/trajectoires
  latentes (par exemple projection PCA) sans prétendre reconstruire des images.
- Ne pas inventer les détails de l'architecture JEPA avant d'avoir défini et testé le
  format séquentiel.

**Validation :** tests de lecture et de frontières, batchs de formes cohérentes, une mise à
jour d'entraînement finie, sauvegarde/rechargement et visualisation latente reproductible.

### F11 — Contrôle avec états latents

À réaliser après une première base JEPA, comme deux expériences séparées :

1. **Planification :** utiliser le prédicteur latent pour simuler/évaluer des séquences
   candidates d'actions et choisir celle dont l'état latent prédit est le plus proche de
   l'objectif, avec une méthode de recherche simple et un budget borné.
2. **Policy avec objectif :** policy prenant `latent_courant` et `latent_objectif`, et
   produisant une distribution d'actions. L'entraîner avec RL et réutiliser l'environnement
   et l'infrastructure de run.

Il faudra définir plus tard comment fournir un état objectif (observation/image cible ou
état issu de l'environnement), comment mesurer la distance latente et gérer les cas où le
modèle ne prédit pas correctement. Il n'est pas demandé de faire du raisonnement
hiérarchique.

**Validation :** protocole comparable, objectifs atteignables définis, métriques de succès
et comparaison séparée des deux approches ; ne pas les présenter comme résolus par la seule
perte JEPA.

## 6. Ordre d'implémentation

Les étapes sont ordonnées par dépendances. Chaque étape doit avoir ses tests ciblés avant
de passer à la suivante.

### Étape 0 — Structure et conventions

- Conserver le paquet Python `gymnasium_wm` existant.
- Définir les conventions de nommage d'IDs, graines, logs, erreurs et chemins.
- Définir le contrat de configuration et créer un exemple minimal CartPole.
- Décider le schéma HDF5 versionné et vérifier sur un rendu réel la forme/dtype des images.

**Sortie :** spécification des formats et un test de création d'environnement, sans
entraînement.

### Étape 1 — Run minimal CartPole

- Créer configuration/validation, identifiant, dossier de run et résumé.
- Ajouter fabrique Gymnasium, gestion de graine et fermeture sûre.
- Ajouter une petite commande terminal pour exécuter une politique aléatoire sur un budget
  limité.

**Sortie :** run traçable et reproductible avec config figée et résumé.

### Étape 2 — Écrivain HDF5

- Écrire images/actions et index d'épisodes ; traiter l'image initiale, les fins d'épisode
  et les dimensions.
- Ajouter lecteur de validation, fermeture/reprise contrôlée et écriture par blocs.
- Ajouter les données de politique aléatoire et d'évaluation dans le même fichier, sans
  identification de leur type.

**Sortie :** HDF5 lisible, frontières testées et taille/temps de stockage mesurés.

### Étape 3 — Infrastructure commune d'apprentissage

- Boucle d'interaction, progression, compteurs, CSV/TensorBoard, évaluation et vidéos.
- Vérifier la distinction des métriques d'entraînement/évaluation dans les résultats, sans
  la reporter dans HDF5.

**Sortie :** expérience de bout en bout avec politique aléatoire.

### Étape 4 — DQN maison

- Réseaux, buffer, exploration, apprentissage et tests unitaires.
- Entraîner un run CartPole court, évaluer, sauvegarder et reprendre.

**Sortie :** DQN fonctionnel avec un checkpoint de reprise vérifié.

### Étape 5 — PPO maison

- Réseaux politique/valeur, rollout, avantages et optimisations PPO.
- Entraîner, évaluer, sauvegarder et reprendre un run CartPole.

**Sortie :** PPO fonctionnel avec le même format de résultats.

### Étape 6 — Références SB3

- Ajouter le lancement de référence et aligner le protocole d'évaluation.
- Produire un tableau/CSV comparatif par run sans masquer seeds et budgets.

**Sortie :** comparaison reproductible des modèles maison et SB3.

### Étape 7 — LunarLander

- Tester rendu, observation, actions et limites de temps.
- Ajouter les configurations DQN/PPO compatibles et tests de fumée.

**Sortie :** second environnement utilisé par les implémentations maison.

### Étape 8 — Mesure puis vectorisation

- Établir benchmark simple.
- Ajouter `SyncVectorEnv` uniquement si les gains mesurés valent la complexité.
- Valider l'indexation des épisodes par environnement vectorisé.

**Sortie :** choix documenté entre instance unique et environnement vectorisé.

### Étape 9 — Autres familles Gymnasium

- Intégrer Toy Text individuellement, avec leurs dépendances, rendus et espaces.
- Intégrer ensuite CarRacing, observer le coût des images et définir le support des actions
  continues pour PPO ; ne pas prétendre que DQN traite directement son espace continu.
- Atari reste une extension future distincte, avec gestion des ROMs et dépendances
  spécifiques.

**Sortie :** matrice de compatibilité des environnements réellement testés.

### Étape 10 — Dataset et premier JEPA

- Finaliser lecteur de séquences et tests inter-épisodes.
- Définir une architecture JEPA de référence pédagogique et son objectif exact.
- Implémenter entraînement, sauvegardes, TensorBoard/CSV et exploration des latents.

**Sortie :** entraînement JEPA sur les données HDF5, sans décodeur d'image.

### Étape 11 — Deux stratégies d'objectif

- Établir une tâche simple et ses états/objectifs mesurables.
- Implémenter un prototype de planification et mesurer son coût.
- Implémenter séparément la policy conditionnée par l'état latent actuel et objectif,
  entraînée par RL.
- Comparer réussite, longueur de trajectoire et robustesse.

**Sortie :** deux expériences isolées, sans hiérarchie ni capacités non demandées.

## 7. Organisation cible des fichiers

Les dossiers ci-dessous sont préparés. Les modules métier seront ajoutés à l'étape où ils
seront implémentés ; les hyperparamètres ne sont pas pré-remplis sans décision/test.

```text
configs/
  settings.yaml
  algorithms/
  environments/
doc/
  architecture.md
  cahier-des-charges.md
src/gymnasium_wm/
  __init__.py
  environments/       fabrique et vérifications des environnements
  runs/               gestion d'un run, résumé et sauvegardes
  storage/            contrat HDF5, writer et reader
  rl/                 composants communs, DQN et PPO
  evaluation/         métriques, évaluateur et vidéo
  world_models/       dataset séquentiel et JEPA
  control/            planning latent et policy avec objectif
  utils/              graines, horodatage et helpers partagés
tests/
  unit/               tests rapides d'une fonction/composant
  integration/        échanges réels Gymnasium, HDF5 et mini-runs
data/
  runs/
  datasets/
```

Responsabilités des modules envisagés :

- `cli.py` : commandes utilisateur, arguments et messages d'erreur.
- `config.py` : chargement, fusion déterministe, validation et snapshot.
- `environments/factory.py` : construction Gymnasium et wrappers explicitement choisis.
- `runs/manager.py` : identifiant, répertoires, statut, résumé et fermeture.
- `storage/hdf5_writer.py` / `hdf5_reader.py` : format versionné et accès séquentiel.
- `rl/networks.py`, `rl/dqn.py`, `rl/ppo.py` : modèles et logique RL lisible.
- `rl/replay_buffer.py` / `rl/rollout_buffer.py` : mémoire propre à chaque algorithme.
- `evaluation/metrics.py` / `evaluation/video.py` : agrégats, CSV/TensorBoard, captures.
- `checkpoints.py` : sauvegarde atomique et reprise contrôlée.
- `world_models/sequences.py` / `world_models/jepa.py` : fenêtres d'épisodes et modèle.
- `control/planning.py` / `control/goal_policy.py` : expériences futures de contrôle latent.
- `utils/seeding.py` : graines des générateurs Python, NumPy, PyTorch et Gymnasium.

Cette liste est une séparation de responsabilités, pas une obligation de créer un fichier
pour chaque helper : garder le projet simple et regrouper les modules si le code reste
lisible.

## 8. Critères d'acceptation de la V1

La V1 RL/collecte est considérée fonctionnelle lorsque :

- une configuration lance un run CartPole reproductible, traçable et CPU-only ;
- le run écrit une configuration figée, des métriques, un résumé, un modèle et les vidéos
  d'évaluation prévues ;
- PPO et DQN maison passent tests unitaires ciblés et peuvent effectuer un entraînement
  complet court sans erreur ;
- un checkpoint permet de reprendre correctement un run compatible ;
- la référence SB3 est comparable sous les mêmes budgets et seeds ;
- les images RGB et actions des épisodes d'entraînement et d'évaluation sont toutes
  collectées dans le HDF5 du run ;
- l'index HDF5 sépare les épisodes, sans tag train/evaluation et sans flags
  `terminated`/`truncated` ;
- le fichier peut être rouvert et lu sans charger tout le dataset en mémoire ;
- LunarLander est supporté avant de déclarer le socle RL multi-environnement utilisable ;
- chaque comportement principal est documenté et couvert par test.

La V1 WM est considérée fonctionnelle lorsque le lecteur fournit des séquences correctes,
le JEPA peut effectuer et reprendre un entraînement CPU, et les représentations latentes
peuvent être inspectées sans décodeur.

Toy Text, CarRacing, vectorisation, planning et policy d'objectif doivent être indiqués
comme non supportés tant que leurs étapes et tests dédiés ne sont pas terminés.

## 9. Points à décider au moment de l'implémentation

Ne pas figer ces détails par supposition ; choisir au moment de l'étape concernée et
documenter la décision :

- valeurs des hyperparamètres et budgets d'entraînement initiaux ;
- structure exacte des fichiers YAML spécifiques et ordre de priorité ;
- politique de nommage des runs et de conservation des anciens checkpoints ;
- dimensions RGB réelles par environnement, recadrage/redimensionnement éventuel ;
- compression HDF5, tailles de blocs et taille des séquences ;
- alignement exact autour du reset et de la dernière image d'un épisode ;
- fréquence et nombre d'épisodes d'évaluation/vidéo ;
- RNG et éléments d'état nécessaires à une reprise exacte sur CPU ;
- architecture JEPA, objectif de perte, cible et horizon ;
- définition d'un objectif latent atteignable et métriques du planning/policy.

Un point important est déjà décidé : les HDF5 initiaux mélangeront les épisodes
d'entraînement et d'évaluation sans identification. Si l'on veut plus tard séparer ces
sources ou mesurer une généralisation sans fuite de données, il faudra modifier le schéma
et/ou refaire une collecte avec une séparation définie.
