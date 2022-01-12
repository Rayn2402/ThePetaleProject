# Plan du repo


## SRC 
Le folder "src" est le plus important du repo. Il contient l'ensemble des outils permettant de réaliser les expériences.

### DATA
Le folder data est divisé lui-même en 2 parties. Il y a la partie "extraction" et la partie "processing".

##### EXTRACTION
Dans le folder "extraction" tu trouveras le fichier *data_management.py*. Celui-ci contient principalement la classe *DataManager* qui aide à interagir avec une base de données PostgreSQL. Ne t'attarde pas nécessairement à la classe *PetaleDataManager* qui si trouve également, celle-ci est spécifique à mon projet. Finalement, tu remarqueras que certaines méthodes du *DataManager* utilises des petites fonctions génériques qui se trouvent dans *helpers.py*. Sinon, le fichier *constants.py* contient une grande quantité de constantes qui sont réutilisées à plusieurs reprises dans mon projet, principalement des noms de tables SQL, ainsi que des noms de colonnes. Ce fichier nécessite un grand ménage car il est un peu en bordel donc ne te concentre pas trop là-dessus.

##### PROCESSING
Dans le folder *processing* tu trouveras une multitude de fichiers ayant des classes et des fonctions permettant de manipuler les données (une fois que celle-ci ont été extraites).

 Le fichier *cleaning.py* contient le classe *DataCleaner* qui fournit la procédure qui permet de faire le nettoyage d'une table extraite de PostgreSQL. Plus précisément, la méthode __call__ du *DataCleaner* regarde un dataframe et retire les lignes et colonnes avec trop de données manquantes, en plus d'identifier des outliers potentiels. Plusieurs figures ainsi qu'un fichier .json avec les détails du nettoyage sont générés par cette méthode.

Le fichier *dataset.py* contient la classe *PetaleDataset* qui est le contenant dans lequel les données d'un *learning set* sont conservés lors d'une expérience. Tu peux te référer à la figure du document *workflow.pdf"* si tu ne te souviens pas de ce que j'entends par *learning set*. La classe *PetaleDataset* est volumineuse et peu paraître complexe, tu peux la voir comme une grosse boîte dans laquelle on met les données. Cette grosse boîte dispose de pleins de méthodes permettant de travailler avec les données. D'ailleurs, certaines méthodes utilises des fonctions qui se retrouvent dans les fichiers *preprocessing.py* et *transforms.py*. 

De son côté, le fichier *sampling.py* contient la classe *RandomStratifiedSampler* qui permet de réaliser plusieurs échantillons stratifiés sur un *PetaleDataset*. En particulier la méthode __call__ du *RandomStratifiedSampler* ne va pas réellement divisé le dataset en plusieurs sous-ensembles, mais plutôt sauvegarder différentes listes *train*, *valid* et *test* contenant des index de patients qui doivent se retrouver respectivement dans l'ensemble d'entrainement, de validation et de test de façons à produire des échantillons stratifiés.

Le fichier *feature_selection.py* contient la classe *FeatureSelector*. Cet objet a été implémenté dans le but d'identifier les features les plus importants dans un dataset. Tu verras son utilisation plus loin, met garde en tête que la façon dont le *FeatureSelector* procède est simplement d'entraîner un modèle de Random Forest et de cibler les meilleurs features à l'aide du *feature importance* calculé par le Random Forest.


### MODELS
Le dossier *models* contient l'ensemble du code lié à la création de modèle de *machine learning* qui peuvent être utilisé pour réalisé des expériences. On retrouve les 3 folders principaux *abstract_models*, *blocks* et *wrappers*.

##### ABSTRACT MODELS
Dans le dossier *abstract_models* tu trouveras plusieurs fichiers qui contiennent des classes abstraites qui ne sont pas directement utilisés dans mon projet. Ce sont plutôt des classes qui représentent des objets simples à partir desquelles ont peu faire des modèles plus complexes qui sont également compatibles avec d'autres modules d'optimisation d'hyperparamètre que nous verrons plus tard.

Le fichier *base_models.py* est simple mais TRÈS important. Il contient les classes *PetaleBinaryClassifier* et *PetaleRegressor*. Ces classes représentent respectivement le squelette commun de l'ensemble des modèles de classification binaire ainsi que les modèles de régression. Autrement dit, tous les modèles que l'on construit doivent **hérités** d'une de ces classes pour être conformes avec le reste du projet. Donc, comme tu peux le constater, mon framework n'est pas encore généralisable a des problèmes de classifications avec plus de 2 classes. Peut-être que cela viendra!

Le fichier *custom_torch_base.py* contient la classe *TorchCustomModel*. Cette classe a été créé de façon à recueillir l'ensemble des méthodes communes aux modèles PyTorch qui sont implémentés dans le projet. Ainsi, les modèles PyTorch crée peuvent hérité de cette classe et simplement implémenter les quelques fonctions manquantes.

Les fichiers *han_base_models.py* et *mlp_base_models.py* contiennent respectivement le code des modèles de classification et de régression associés aux modèles **H**eterogeneous Graph **A**ttention **N**etwork (HAN) et **M**ulti-**L**ayer **P**erceptron (MLP). À ce moment, tu peux te poser la question : "Pourquoi mettre ces modèles dans le *abstract_models* s'ils sont fonctionnels?". En fait, bien qu'ils sont fonctionnels, ils n'héritent pas encore des classes *PetaleBinaryClassifier* et *PetaleRegressor* donc ils ne sont pas encore compatibles avec le reste des modules du packages. C'est pourquoi je les considères abstraits. 

Finalement, tu peux jeter un coup d'oeil au fichier *encoder.py*, mais il n'est pas très important.

##### BLOCKS
Le dossier *blocks* ne contient que des fichiers comportant des blocs d'architectures de réseaux de neurones. Le fichier *mlp_blocks.py* contient des blocs qui se retrouvent principalement dans des modèles MLP tandis que le fichier *gnn_blocks* contient des blocs qui se trouvent dans des **G**raph **N**eural **N**etwork (GNN).

##### WRAPPERS
Le dossier *wrappers* contient le code qui permet de faire le pont entre les modèles abstraits et le reste des modules implémentés dans le projet.

En gros, le fichier *sklearn_wrappers.py* contient les classes *SklearnBinaryClassifierWrapper* et *SklearnRegressorWrapper*. Le but de ces classes est d'offrir un genre d'enveloppe dans lequel on peut mettre des modèles avec un API semblables aux modèles offerts par la librairie *Scikit Learn (sklearn)* et les utiliser dans le projet Petale. Plus clairement, si j'ai en main un modèle déjà existant avec un API comme celui de *sklearn* et que je souhaite qu'il soit compatible avec le reste des modules du projet, je peux le mettre dans un *wrapper* et le tour est joué! Les *wrappers* vont disposés des méthodes nécessaires pour rendre les modèles compatibles avec tout le reste du code.

Le fichier *torch_wrappers.py* quant à lui a la même utilité, mais agit en tant qu'enveloppe pour tous les modèles qui ont été créés en utilisant la classe abstraite *TorchCustomModel*.

##### AUTRES...
Les fichiers restants qui sont dans aucun dossier contiennent les modèles qui peuvent être utilisés dans les expériences. Tu peux les voirs comme des fichiers comportant des modèles qui ont été enveloppés (*wrapped*) de façon a être compatible avec le reste du code.


### UTILS
Le folder *utils* contient plein de petit bout de codes aidant le fonctionnement de plus gros morceaux du projet! Comme un peu le fichier *helpers.py* contient des fonctions utiles pour le *DataManager*. D'ailleurs, je crois que le fichier *helpers.py* devrait éventuellement être bougé dans le dossier *utils* pour faire plus de sens!

Pour éviter de perdre du temps, je te propose de focuser sur les fichiers importants *hyperparameters.py* et *score_metrics.py*.

Dans le fichier *hyperparameters.py*, tu trouveras plein de classe qui permettre de définir ce qu'est un hyperparamètre pour un modèle de *machine learning*. Tu trouveras des classes décrivant ce qu'est un hyperparamètre catégorique *CategoricalHP*, un hyperparamètre numérique discret (entier) *NumericalIntHP* et un hyperparamètre numérique continu (réel) *NumericalContinuousHP*. De leurs côtés, les énumérations *Distribution* et *Range* offrent du vocabulaire pour définir l'espace dans lequel ces hyperparamètres se trouvent.  Tous ces classes seront utilisés lorsque viendra le temps de faire de l'optimisation d'hyperparamètres! Tu peux déjà voir un peu comment ils sont utilisés si tu regarde les différents modèles contenus dans le dossier *src/models*. Chaque modèles a une énumération qui décrit ses hyperparamètres.

De son côté, le fichier *score_metrics.py* contient une grande variété de métriques de performances pour l'ensemble des modèles. Ainsi nous pouvons les comparer et conclure si un modèle est meilleur que d'autres pour une certaine tâche.


### TRAINING
Le folder *training* contient l'ensemble des fichiers qui permettent de réaliser une même expérience avec plusieurs modèles et comparer ainsi leur performance. Nous verrons chacun de ces fichiers en détails.

Le fichier *evaluation.py*  contient la classe *Evaluator*. Ce module s'occupe de prendre un modèle et d'en faire une évaluation pour une tâche et un dataset donné. En gros, la méthode *evaluate* de l'*Evaluator* réalise l'ensemble des actions qui se trouvent sous la boîte *Learning set* dans la figure à la page 2 du document *workflow* que je t'ai transmis. Pour faire un petit résumé de ces étapes, pour plusieurs splits d'entrainement et de test différents, l'*Evaluator* va réaliser une optimisation d'hyperparamètres en utilisant l'ensemble d'entraînement et va tester le modèle avec les meilleurs hyperparamètres sur l'ensemble de test.  Plusieurs métriques de performance seront calculées à chaque split, et nous pourrons avoir les moyennes et écart types de celles-ci une fois l'ensemble de l'évaluation terminé.

Le fichier *tuning.py* contient 2 classes importantes, soient *Objective* et *Tuner*. Le *Tuner* est un objet qui utilise les fonctions d'une librairie qui s'appel *Optuna* pour lancer une recherche d'hyperparamètres. Le fonctionnement du *Tuner* est simple. Son but est de donner un ensemble d'hyperparamètres à une fonction *Objective* et d'avoir en retour un score (voir Appendix D du document *workflow.pdf*). Si la fonction *Objective* doit être minimiser, le *Tuner* tentera à l'aide d'une méthode appelée le *TPE* de trouver les hyperparamètres qui permettre de minimiser la fonction *Objective*. Au contraire si la fonction doit être maximiser, le *Tuner* tentera de trouver l'ensemble d'hyperparamètres qui maximise la fonction. Ce qui est intéressant c'est que le *Tuner* n'a aucune idée de ce qui se passe dans la fonction *Objective*, il ne sait que ce que il lui donne et ce qui est retourné comme score.
Au fil des essais (*trials*), il est de mieux en mieux pour trouver des hyperparamètres qui donnent de bon résultats! Pour sa part, la fonction Objective a été conçu de façon a fonctionner pour n'importe quel modèle suivant le cadre d'un *PetaleBinaryClassifier* ou un *PetaleRegressor*, auquel on a définit les hyperparamètres. Toutefois, note que pour que le *Tuner* connaisse l'espace de recherche associé à chaque hyperparamètre dans lequel il peut échantilloner, un dictionnaire décrivant l'espace de recherche doit lui être fournit. Pour avoir une idée de la forme de ce dictionnaire, jète un coup d'oeil au fichier *hps/warmup_hps.py*.

Les fichiers *early_stopping.py* et *sam.py* sont un peu moins volumineux. Tu y trouveras respectivement les classes *EarlyStopper* permettant de réaliser du early stopping avec les réseaux neuronaux et la classe *Sam* qui est en fait un *optimizer* PyTorch permettant de faire de l'entrainement par descente de gradient en utilisant le concept de *Sharpness-Aware Minimization (SAM)*. Une référence vers l'artice du *Sharpness-Aware Minimization (SAM)* est donné dans l'entête du fichier *sam.py*.

### Recording
Le dossier *recording* contient l'ensemble du code permettant de sauvegarder les détails d'une expérimentation ainsi que ses résultats.

Principalement, dans le fichier *recorder.py* tu trouveras la classe *Recorder* qui permet de sauvegardé tout plein d'information sur une expérimentation et qui est d'ailleurs utilisé par l'*Evaluator* que tu as déjà vu. En particulier, lorsqu'un *Evaluator* réalise une grande expérience, il sauvegarde plusieurs données à chaque split d'entraînement et de test à l'aide du *Recorder*. Tu pourra constater aussi qu'à la fin de la fonction *evaluate* du l'*Evaluator*, la fonction *get_evaluation_recap* du fichier *recorder.py* est appelé pour crée un document *.json* dans lequel un résumé des résultats de l'ensemble des splits se trouvent.






