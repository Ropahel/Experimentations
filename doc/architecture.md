# Organisation des runs et des datasets

## Un run, un dossier de résultats

Chaque run entraîne un modèle et reçoit un identifiant unique. Ses résultats sont rangés
sous `data/runs/<run_id>/`. Ce dossier contient les informations nécessaires pour
comprendre, évaluer et éventuellement reprendre le run :

- configuration figée réellement utilisée ;
- résumé lisible et statut du run ;
- métriques d'entraînement et d'évaluation (CSV et TensorBoard) ;
- vidéos d'évaluation périodiques ;
- checkpoints et modèle final ;
- chemin vers le HDF5 associé.

Les dossiers de résultats restent dans `data/runs/`. Les données brutes HDF5 n'y sont pas
stockées.

## Un HDF5 par run et environnement

Le chemin racine des datasets est configurable dans `configs/settings.yaml`. Par défaut,
le fichier d'un run est rangé à l'emplacement suivant :

```text
<datasets_dir>/<environment_id>/<run_id>.h5
```

Un run utilise un environnement à la fois. Si plusieurs runs du même environnement sont
réalisés, chacun garde son propre HDF5 ; les fichiers ne sont pas fusionnés avec ceux des
autres runs. Les données d'un environnement ne sont donc pas mélangées avec celles d'un
autre environnement.

## Contenu et limites des épisodes

Le fichier conserve les images RGB rendues et les actions, à chaque transition. Les
récompenses et les poids du modèle restent dans les résultats du run. Les images initiales
et finales et les actions doivent être alignées sans ambiguïté. Pour un épisode comprenant
T actions, le schéma pourra enregistrer T+1 images : une image initiale et une image après
chaque action.

Un index d'épisodes dans le HDF5 permet de retrouver les plages d'images et d'actions
associées à chaque épisode. Il ne faut pas créer un fichier permanent par épisode. Le
format initial ne distingue pas les épisodes d'entraînement de ceux d'évaluation et ne
stocke pas les flags Gymnasium `terminated` et `truncated`. L'index de frontières reste
toutefois indispensable pour que le chargeur JEPA ne construise pas de séquence traversant
deux épisodes.

Les images et actions sont les données brutes destinées au futur world model. Les
identifiants de run/environnement et les bornes d'épisodes sont des métadonnées
structurelles nécessaires pour interpréter ces données, pas des labels de type de run.

## Écriture et vectorisation

La première version collecte les données avec une seule instance d'environnement. Les
évaluations ajoutent aussi leurs images/actions au HDF5, sans étiquette particulière.

Une éventuelle vectorisation avec `gymnasium.vector.SyncVectorEnv` sera ajoutée après
mesure. Plusieurs environnements vectorisés peuvent transmettre leurs transitions à un
écrivain unique qui les ajoute séquentiellement au même HDF5. Cela évite les écritures
simultanées dans un fichier HDF5 et ne nécessite pas de créer/fusionner des fichiers
temporaires par instance. Un épisode reste indexé séparément pour chaque instance
vectorisée.

## Source des images et compatibilité

La source prévue est `env.render()` avec `render_mode="rgb_array"`. Le collecteur devra
vérifier que chaque environnement expose effectivement une image RGB, en contrôler la
forme et le type, puis enregistrer les images sans sous-échantillonnage au départ. La
résolution et la compression HDF5 seront fixées lors de l'implémentation du collecteur,
après inspection des rendus réels.
