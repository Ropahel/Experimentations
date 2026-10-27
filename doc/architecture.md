# Organisation des runs et des datasets

## Un run, un dossier de résultats

Chaque entraînement aura son propre dossier sous `data/runs/`, identifié par un
identifiant unique. Il pourra contenir la configuration réellement utilisée, les
métriques (dont les pertes et récompenses), les poids du modèle et les vidéos. Un run ne
possède pas son propre dataset brut.

## Un dataset partagé par environnement

Les datasets seront rangés sous `data/datasets/<environment_id>/`. Tous les runs du même
environnement alimenteront le même dataset ; des environnements différents auront des
dossiers distincts. Le contenu HDF5 est réservé aux images et actions brutes pour
entraîner des représentations du monde. Les récompenses, pertes, poids et vidéos restent
dans les résultats du run.

Les environnements Gymnasium n'exposent pas tous leurs images de la même façon. Pour les
environnements rendables, `render_mode="rgb_array"` permet généralement de récupérer une
image depuis `env.render()`. Certains environnements fournissent aussi des observations
visuelles. Il faudra vérifier la source d'image et le format pour chaque famille
d'environnement avant de collecter ses données.

## Choix à faire avant d'écrire le collecteur

- Stocker un fichier HDF5 par épisode ou ajouter les épisodes à un fichier plus gros.
- Définir les formes et types exacts des images et actions pour chaque environnement.
- Empêcher les écritures concurrentes dans un même fichier HDF5 si plusieurs runs se
  terminent en même temps.
- Choisir comment représenter les fins d'épisode sans mélanger les données de plusieurs
  environnements.

Commencer avec un fichier par épisode est une option simple à explorer ; le format final
sera choisi lors de l'implémentation du collecteur. Ce dépôt ne crée pas encore ces
fichiers et ne lance pas d'entraînement.
