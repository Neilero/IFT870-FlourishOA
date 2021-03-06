# %%
"""
# IFT870 - Examen intratrimestriel

Auteur : Aurélien Vauthier (19 126 456)
"""

# %%
# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from difflib import SequenceMatcher
from tqdm import tqdm

# extract data
journal = pd.read_csv("api_journal11-13-17.csv", encoding="latin1")
price = pd.read_csv("api_price11-13-17.csv", index_col=0)
influence = pd.read_csv("estimated-article-influence-scores-2015.csv", index_col=0)

# %%
"""
## Question 1 : Exploration-Description

*Présenter une description de chacun des attributs des 3 tables, avec des graphiques
pour la visualisation des statistiques descriptives au besoin.*
"""

# %%
"""
### Table `journal`
"""

# %%
# show the first values
journal.head()

# %%
"""
Nous pouvons déjà observer les colonnes suivantes et imaginer une petite description : 
- **issn** (valeur numérique) : ([International Standard Serial Number](https://fr.wikipedia.org/wiki/International_Standard_Serial_Number)),
il s'agit d'un numéro permettant d'identifier une série de publication de façon unique.
- **journal_name** (valeur catégorique) : Le nom du journal
- **pub_name** (valeur catégorique) : Le nom de l'éditeur
- **is_hybrid** (valeur booléenne) : (D'après le site [FlourishOA](http://flourishoa.org/about#type)) Permet de savoir
si le journal est hybride. C'est-à-dire, si le journal est à abonnement avec certains articles en accès libre.
- **category** (valeur catégorique) : La liste des catégories de la revue scientifique.
- **url** (valeur catégorique) : L'adresse web de la page d'acceuil du journal
"""

# %%
#  Compute ratio of N/A values
(journal.isna().sum() / journal.shape[0]) * 100

# %%
"""
Nous pouvons remarquer que les colonnes `pub_name`, `category` et `url` possèdent des données manquantes. En particulier
`category` et `url` qui ont environ 50% de données manquantes.
"""

# %%
# Compute ratio of unique values
print(f"Ratio de valeurs uniques pour l'index : {len(np.unique(journal.index)) / journal.shape[0]:.0%}")
print(f"Ratio de valeurs uniques pour issn : {journal['issn'].nunique() / journal.shape[0]:.0%}")

# %%
"""
L'index et les ISSN sont bien uniques.

La colonne `journal_name` ne semble pas avoir de problème mis à part quelques données dupliquées.
"""

# %%
# sort and print the 10 biggest publishers
pub, count = np.unique(journal["pub_name"].dropna(), return_counts=True)
print("Nombre de revue par éditeur :")
for pub, count in sorted(zip(pub.tolist(), count.tolist()), key=lambda x: x[1], reverse=True)[:10]:
    print(f"\t- {pub} : {count} ({count/journal.shape[0]:.2%})")

# %%
"""
On peut voir qu'il existe de gros éditeurs, en particulier `Springer` qui publie 14.3% des revues scientifiques. 
"""

# %%
# convert to bool values
journal["is_hybrid"] = journal["is_hybrid"].astype(bool)
print(f"Ratio de revues hybrides : {journal['is_hybrid'].sum() / journal.shape[0]:.2%}")

# %%
journal["category"][11065:11070]

# %%
"""
Comme on peut le voir sur l'exemple ci-dessus, certaines revues possèdent plusieurs catégories qui peuvent être
séparées par plusieurs caractères comme `|`, `.` et `and`.

Finalement, pour la colonne `url`, il y a une grande partie de données manquantes.
"""

# %%
# show stats about categories
categories, cat_counts = np.unique(journal["category"].dropna(), return_counts=True)
categories = np.delete(categories, np.where(cat_counts < 100))
cat_counts = np.delete(cat_counts, np.where(cat_counts < 100))
sns.barplot(x=categories, y=cat_counts)
plt.xticks(rotation=90)
plt.show()

# %%
"""
Comme on peut le voir, certaines catégories sont bien plus représenté que d'autres. De plus, certaines catégories sont
dupliquées avec des polices différentes (comme `Medicine` et `MEDICINE`).
"""

# %%
"""
### Table `price`
"""

# %%
# show the first values
price.head()

# %%
"""
Nous pouvons déjà observer les colonnes suivantes et imaginer une petite description :
- **price** (valeur continue) : Le prix de l'ACP (Article Publication Charge)
- **date_stamp** (valeur temporelle) : Horodatage représentant la date de création de l'entrée.
- **journal_id** (valeur catégorique) : Il s'agit de l'ISSN du journal
- **influence_id** (valeur catégorique) : On pourrait supposer qu'il s'agit d'un lien vers les lignes de la table
`influence` mais la majorité des id situés dans cette colonne sont supérieurs au nombre de lignes que possède la table
`influence` ce qui constitue alors des valeurs aberrantes.
- **url** (valeur catégorique) : L'adresse web de la revue vers la page d'information pour les auteurs.
- **license** (valeur catégorique) : Valeur numérique représentant une licence (nous n'avons pas d'information sur la
correspondance entre les valeurs numériques et les différentes license qui existent.
"""

# %%
#  Compute ratio of N/A values
(price.isna().sum() / price.shape[0]) * 100

# %%
"""
Nous pouvons remarquer que les colonnes `influence_id`, `url` et `license` possèdent presque uniquement des données
manquantes.
"""

# %%
def show_continuous_col_stats(df, column):
    print(f"Statistiques de la colonne {column} :")
    print(f"- min : {df[column].min():.2f}")
    print(f"- max : {df[column].max():.2f}")
    print(f"- moyenne : {df[column].mean():.2f}")
    print(f"- varience : {df[column].var():.2f}")
    print(f"- mode : {df[column].mode()[0]:.2f}")
    sns.distplot(df[column].dropna())

show_continuous_col_stats(price, "price")

# %%
"""
On remarque sur le graphique, 4 groupes de prix des journaux. Un premier à 0, un deuxième entre 100 et 2900 environ, un
troisième autour de la valeur 3000 et finalement un dernier groupe au-dessus de 3100.
"""

# %%
price["date_stamp"] = pd.to_datetime(price["date_stamp"], format="%Y-%m-%d")
groupedPrice = price["date_stamp"].groupby(price["date_stamp"].dt.year, sort=False)
years, years_count = [], []
for year in groupedPrice:
    years.append(year[0])
    years_count.append(year[1].shape[0])

plt.hist(years, bins=len(years), weights=years_count)
plt.show()

# %%
"""
On note sur le graphique que la plupart des données ont été ajoutée en 2016 et 2017, une autre partie fut ajouté en 2012
et 2013. La majorité des données est donc assez récente.
"""

# %%
def multicolumn_duplicate_ratio(df, columns):
    flatten_array = df[columns].values.ravel('K')
    n_unique = len(pd.unique(flatten_array))
    return n_unique / df.shape[0]


# Compute ratio of unique values
print(f"Ratio de valeurs uniques pour journal_id : {price['journal_id'].nunique() / price.shape[0]:.0%}")
print(f"Ratio de valeurs uniques pour (journal_id, date_stamp) : {multicolumn_duplicate_ratio(price, ['date_stamp', 'journal_id']):.0%}")

# %%
"""
Contrairement à la table `journal` il existe plusieurs duplicatas des ISSN, en effet, certaines revues scientifiques
ont eu des mises à jours de leurs informations.
"""

# %%
# show ratio of incoherent values
print(f"Ratio de valeurs incohérentes pour influence_id : {1 - (price['influence_id'] <= influence.index[-1]).sum() / price.shape[0]:.2%}")

# %%
"""
La quasi-totalité des valeurs de `influence_id` sont soit manquantes soit ont une valeur supérieur à la valeur maximale
des id de la table `influence`. Cette colonne ne semble donc pas être utile.
"""

# %%
print("Valeur(s) abérente(s) pour la colonne url :")
incoherent_influence_url = price[~price["url"].str.startswith("http", na=True)]
for index, row in incoherent_influence_url.iterrows():
    print(f"{index} : {row['url']}")

# remove incoherent values
for index in incoherent_influence_url.index:
    price.at[index, "url"] = np.NaN

# %%
"""
Bien qu'on ne retrouve qu'une seule valeur aberrante, la majorité des valeurs reste manquante. Cela nuit donc fortement
à l'intérêt de cette colonne.
"""

# %%
licenses, licenses_count = np.unique(price["license"].dropna(), return_counts=True)
plt.hist(licenses, bins=len(licenses), weights=licenses_count)
plt.show()

# %%
"""
Comme nous pouvons le voir sur l'histogramme, la licence 2 est majoritairement utilisée, on retrouve ensuite les
licences 10, 4 et 6. Cependant, ne pouvant faire l'équivalence entre ces numéros et le nom des licences (ou groupes de
licences), les informations de cette colonne ne sont pas pertinentes. 
"""

# %%
"""
### Table `influence`
"""

# %%
# show the first values
influence.head()

# %%
"""
Nous pouvons déjà observer les colonnes suivantes et imaginer une petite description :
- **journal_name** (valeur catégorique) : Le nom du journal.
- **issn** (valeur catégorique) : L'ISSN du journal.
- **citation_count_sum** (valeur continue) : Le nombre de citations du journal.
- **paper_count_sum** (valeur continue) : Le nombre d'articles scientifiques du journal.
- **avg_cites_per_paper** (valeur continue) : La moyenne du nombre de citations par article du journal.
- **proj_ai** (valeur continue) : Le score d'influence associé à la moyenne des citations.
- **proj_ai_year** (valeur temporelle) : La date associée au calcul du score d'influence.
"""

# %%
#  Compute ratio of N/A values
(influence.isna().sum() / influence.shape[0]) * 100

# %%
"""
Il n'y a presque aucune données manquante dans la table `influence`. De plus, les quatre seules colonnes en possédant
un peu, `citation_count_sum`, `paper_count_sum`, `avg_cites_per_paper` et `proj_ai`, ont exactement le même nombre de
données manquantes : 0.36%.

La colonne `journal_name` ne semble pas avoir de problème mis à part quelques données dupliquées.
"""

# %%
# Compute ratio of unique values
print(f"Ratio de valeurs uniques pour l'index : {len(np.unique(influence.index)) / influence.shape[0]:.0%}")
print(f"Ratio de valeurs uniques pour issn : {influence['issn'].nunique() / influence.shape[0]:.0%}")

# %%
"""
L'index et les ISSN sont bien uniques.
"""

# %%
show_continuous_col_stats(influence, "citation_count_sum")

# %%
"""
On peut noter un très grande dispersion des valeurs du nombre de citations qui peut faire pense à une distribution de
Poisson. De nombreux journaux n'ont pas beaucoup de citation (environ 636) alors que certains journaux se dispersent
entre des valeurs de 10 000 à 430 000 citations.
"""

# %%
show_continuous_col_stats(influence, "avg_cites_per_paper")

# %%
"""
Là encore la similarité avec la loi de Poisson est visible mais cette fois-ci la dispersion des données est bien moins
présente. On note tout de même un pic autour de la valeur 1.75 et des valeurs allant jusqu'à 27. 
"""

# %%
show_continuous_col_stats(influence, "proj_ai")

# %%
"""
Cette colonne étant le résultat d'un rapport entre les deux dernières colonnes, il n'est pas très surprenant d'observer
une dernière fois la loi de Poisson avec un mode autour de la valeur 0.4 et une dispertion jusqu'à 11.
"""

# %%
print("Liste des valeurs uniques de proj_ai_year :")
for year in influence["proj_ai_year"].unique():
    print(f"\t- {year}")

# %%
"""
Surprenamment, nous pouvons noter que seule l'année 2015 est présente dans cette colonne.
"""

# %%
"""
## Question 2 : Prétraitement-Représentation

*Effectuer un prétraitement des données pour supprimer les duplications et corriger les
incohérences s’il y en a.*
"""

# %%
"""
### Table `journal`

On pourrait supposer que les ISSN permettent de définir une revue de façon unique, cependant, il existe de nombreuses
lignes dupliquées qui possèdent exactement les mêmes informations hormis leur ISSN. Il nous faut donc trouver un 
sous-ensemble de colonne qui nous permettrons de définir des duplicatas. 

Une deuxième hypothèse que nous pouvons faire est qu'un journal est représenté par son nom et qu'il est peu probable que
deux journaux différents possède le même nom. Nous allons donc considérer que deux lignes possédant le même nom de
journal sont des duplicatas. Pour différencier deux deuplicatas nous allons ensuite calculer un poids correspondant au
nombre de valeurs non manquantes + 5 si la colonne `category` est non manquante. Ce choix de privilégier la colonne
`category` est fait de façon à privilégier les lignes avec cette colonne car elle sera importante pour les prédictions
des questions suivantes.
"""

# %%
# Clean string columns
journal["issn"] = journal["issn"].str.strip()
journal["journal_name"] = journal["journal_name"].str.strip()
journal["pub_name"] = journal["pub_name"].str.strip()
journal["category"] = journal["category"].str.strip().str.lower()
journal["url"] = journal["url"].str.strip()

# remove duplicated values
not_na_count = journal.notnull().sum(axis=1)
row_to_keep = not_na_count.mask(journal["category"].notna(), not_na_count+5).groupby(journal["journal_name"]).idxmax()
journal = journal.loc[row_to_keep]

print(f"Nombre de lignes condérées comme dupliquées supprimées : {not_na_count.shape[0] - row_to_keep.shape[0]}")

# Replace inconsistant seperators by one so that we can seperate the values easily later
journal["category"] = journal["category"].str.replace(r"\s*([|.,&]|and)\s*", '|', regex=True)

# %%
"""
### Table `price`

Cette table étant une liste horodatée de prix pour un journal, nous pouvons donc utiliser le couple des colonnes 
`date_stamp` et `journal_id` pour chercher les duplicatas.
"""

# %%
# Clean string columns
price["journal_id"] = price["journal_id"].str.strip()
price["url"] = price["url"].str.strip()

# Check duplicates
price[price.duplicated(subset=["date_stamp", "journal_id"]) & ~price.duplicated(subset=["date_stamp", "journal_id", "price"])]

# %%
"""
Seul un article possède un prix différent pour la même date (id `13073` et `16473`). En visitant le [site de la revue](https://jpl.letras.ulisboa.pt/about/submissions/)
on peut trouver la mention de *publiction fee* de £330. On peut donc considérer la deuxième ligne avec un prix affiché de
$387.15 comme étant la bonne ligne.
"""

# %%
# Delete the wrong article found
price.drop(13073, inplace=True)

# Delete the other duplicates
price.drop_duplicates(subset=["date_stamp", "journal_id"], inplace=True)

# %%
"""
Nous avions aussi trouvé une URL incohérente lors de la question 1 et l'avons déjà supprimé à ce moment-là.

De plus, nous avions aussi trouvé de très nombreuses valeurs incohérentes dans la colonne `influence_id`. Par conséquent,
nous allons la supprimer. 
"""

# %%
price.drop("influence_id", axis=1, inplace=True)

# %%
"""
### Table `influence`

Pour cette table, à l'instar de `journal`, nous ne pouvons pas nous baser sur la colonne ISSN car plusieurs lignes sont
identiques si on exclu la vérification de l'ISSN.
"""

# %%
# clean string columns
influence["journal_name"] = influence["journal_name"].str.strip()
influence["issn"] = influence["issn"].str.strip()

# drop duplicates
influence.drop_duplicates(subset=influence.drop("issn", axis=1).columns, inplace=True)

# %%
"""
Finalement, la colonne `url` étant entièrement vide, il nous semble inutile de la garder. De plus, la colonne
`proj_ai_year` ne possède qu'une seule valeur non nulle, il nous semble donc peu utile de la garder aussi.
"""

# %%
influence.drop("proj_ai_year", axis=1, inplace=True)

# %%
"""
*Y a t il une corrélation entre les catégories de journaux (attribut « category ») et les
coûts de publication (attribut « price ») ? Justifier la réponse.*
"""

# %%
"""
Afin d'obtenir une corrélation plus précise nous pouvons essayer de ne garder qu'une ligne pour chaque journal. Ce
faisant nous gardons les prix les plus récents afin d'obtenir des statistiques plus à jour.
"""

# %%
# merge tables with the desired columns
merge = journal.merge(price, left_on="issn", right_on="journal_id")

# sort by date stamp
merge.sort_values(by=["date_stamp"], ascending=False, inplace=True)

# only keep the first line of the duplicated journal name
merge.drop_duplicates(subset=["journal_name"], keep="first", inplace=True)

# drop unwanted columns
merge.drop(merge.drop(["category", "price"], axis=1).columns, axis=1, inplace=True)

# drop row where either the price or the category is missing
merge.dropna(inplace=True)

merge = pd.concat([merge, merge["category"].str.get_dummies()], axis=1)

correlations = []
for col in merge.drop(["category", "price"], axis=1).columns:
    group = merge[merge[col] == 1]
    if group.shape[0] >= 10:    # filter out the category with not much data
        correlations.append([col, group["price"].values])

# sort by prices' mean and extract values to list of cols and list of values
correlations = list(zip(*sorted(correlations, key=lambda corr: corr[1].mean(), reverse=True)))

sns.barplot(data=correlations[1])
plt.xticks(plt.xticks()[0], labels=correlations[0], rotation=55, ha="right")
plt.show()

# %%
"""
Comme nous pouvons le voir sur le graphique ci-dessus, il semble que les catégories avec les prix les plus élevés soient
sembles être liées aux différentes catégories des sciences (médecine, biologie, phisique, chimie...). A contrario, il
semble que les catégories avec les plus faibles prix soient tournées autour des arts, de la politique, de la litérature,
de l'histoire...

En résumé, nous pouvons dire que dans le cadre de nos données il existe une corrélation entre les colonnes `price` et
`category`.
"""

# %%
"""
*Construire un modèle pour prédire les valeurs de catégorie de journaux manquantes de
la façon la plus précise possible (cela inclut la sélection d’attributs informatifs, le
choix et le paramétrage d’un modèle de classification, le calcul du score du modèle,
l’application du modèle pour prédire les catégories manquantes). Justifier les choix
effectués.*
"""

# %%
"""
Pour prédire les valeurs des catégories nous allons utiliser les différentes statistiques présentes dans `influence` et
les prix présents dans `price`. Nous n'utiliserons cependant pas les données de `date_stamp` et de `is_hybrid` car ses
données ont plus un lien avec le prix qu'avec les catégories. Nous allons aussi calculer la distance entre les
catégories et les deux premières colonnes de `journal` (`journal_name` et `pub_name`). Pour cette distance, nous allons
calculer la longueur de la sous chaine commune la plus longue et la diviser par la taille de la catégorie afin d'obtenir
un "pourcentage de ressemblance".

Pour le modèle, nous allons utiliser un `MultiOutputClassifier` (pour pouvoir prédire plusieurs catégories à un journal)
avec un `RandomForestClassifier` (pour bénéficier de la capacité des arbres décisionnels et de leur simplicité). Enfin,
pour les hyperparamètres, nous allons utiliser un `GridSearchCV`.
"""

# %%
# merge all data frames
merge = journal.merge(price, left_on="issn", right_on="journal_id").merge(influence, on="issn")
merge.rename(columns={"journal_name_x": "journal_name"}, inplace=True)

# drop unwanted columns
desired_cols = ["journal_name", "pub_name", "price", "citation_count_sum", "paper_count_sum", "avg_cites_per_paper",
                "proj_ai", "category"]
cat_model_data = merge.drop(merge.drop(desired_cols, axis=1).columns, axis=1)

# drop lines missing data used for prediction
desired_cols.remove("category") # we want to predict the missing categories at the end
cat_model_data.dropna(subset=desired_cols, inplace=True)

# compute training data targets
train_mask = cat_model_data["category"].notna()
cat_targets = cat_model_data[train_mask]["category"].str.get_dummies(sep='|')
cat_model_data.drop("category", axis=1, inplace=True)

# lower journal_name strings to make the string distance ignore cases
cat_model_data["journal_name"] = cat_model_data["journal_name"].str.lower()

# define the distance between a string (journal_name or pub_name) and a category
def category_dist(row, base_col, category):
    name = row[base_col]
    if name is np.NaN:
        return 0
    len_match = SequenceMatcher(a=name, b=category).find_longest_match(0, len(name), 0, len(category)).size
    return len_match / len(category)

# compute the journal_name and pub_name distances with the categories for the data
apply_category_dist = lambda df, base, cat: df.apply(category_dist, axis=1, base_col=base, category=cat)
for category in tqdm(cat_targets.columns, desc="Computing distances between (journal_name, pub_name) and the categories"):
    cat_model_data[f"journal_name_to_{category}_dist"] = apply_category_dist(cat_model_data, "journal_name", category)
    cat_model_data[f"pub_name_to_{category}_dist"] = apply_category_dist(cat_model_data, "pub_name", category)

# drop string columns
cat_model_data.drop(["journal_name", "pub_name"], axis=1, inplace=True)

# %%
# define the hyper-parameter's grid search
param_grid = {
    "estimator__max_depth" : np.linspace(13, 15, 3, dtype=int),
    "estimator__n_estimators" : np.linspace(100, 200, 3, dtype=int)
}

# Create the model
rfc = RandomForestClassifier(n_jobs=-1)
moc = MultiOutputClassifier(rfc, n_jobs=-1)
cat_model = GridSearchCV(moc, cv=2, param_grid=param_grid, n_jobs=-1, verbose=1)

# Train model
X_train, X_test, y_train, y_test = train_test_split(cat_model_data[train_mask], cat_targets, test_size=0.2)
cat_model.fit(X_train, y_train)
print(f"Train accuracy : {cat_model.score(X_train, y_train):.2%}")
print(f"Test  accuracy : {cat_model.score(X_test, y_test):.2%}")
print(f"Best params : {cat_model.best_params_}")

# %%
# Predict missing categories
predicted_cat = pd.DataFrame(cat_model.predict(cat_model_data), index=cat_model_data.index)

# show stats about predicted data
sns.barplot(x=cat_targets.columns, y=predicted_cat.sum(axis=0))
plt.xticks(rotation=90)
plt.show()

# replace line where the category in know by its representation in one hot
predicted_cat.mask(train_mask, cat_targets, inplace=True, axis=0)

# add the resulting data to the merge data
predicted_cat.reindex(merge.index)
predicted_cat.fillna(0.)    # these are the categories we could not predict because of missing data
merge = pd.concat([merge, predicted_cat], axis=1)

# %%
"""
Comme on peut le voir sur le graphique la répartition des catégories prédites n'est pas consistante. On remarque en
particulier `medicine` qui est bien plus prédite que le reste et certaines catégories ne semble pas être prédites une
seule fois. Ce résultat pouvait être attendu, car comme nous l'avons vu dans la question 1, `medicine` est la catégorie
la plus représentée. Malgré cela, le modèle obtient quand même un bon score de généralisation / de test.
"""

# %%
"""
## Question 3 : Régression-Clustering

*Supprimer tous les attributs ayant plus de 50% de données manquantes.*
"""

# %%
def drop_empty_columns(df, threshold=0.5):
    df.drop(df.columns[(df.isna().sum() / df.shape[0]) > threshold], axis=1, inplace=True)

for df in [journal, price, influence]:
    drop_empty_columns(df)

# %%
"""
*Construire un modèle pour prédire le coût actuel de publication (attribut « price ») à
partir des autres attributs (cela inclut la sélection d’attributs informatifs, le choix et le
paramétrage d’un modèle de régression, le calcul du score du modèle, l’application du
modèle pour prédire les coûts). Justifier les choix effectués.
Lister les 10 revues qui s’écartent le plus (en + ou -) de la valeur prédite.*
"""

# %%
"""
Pour calculer les coûts actuels de publication, nous allons encore une fois utiliser les différents attributs de la
table `influence` et de `price`. À cela, nous allons aussi utiliser les catégories présentes dans `category` que nous
compléterons avec le modèle précédemment entrainé.

Pour le modèle de regression, nous allons cette fois-ci utiliser un `RandomForestRegressor` avec un `GridSearchCV` pour
la recherche des hyperparamètres.
"""

# %%
# drop unwanted columns
desired_cols = ["citation_count_sum", "paper_count_sum", "avg_cites_per_paper", "proj_ai", "is_hybrid", "price",
                "date_stamp"] + list(range(predicted_cat.shape[1])) # the one hots of the categories
price_model_data = merge.drop(merge.drop(desired_cols, axis=1).columns, axis=1)

# only keep the year for the date stamp
price_model_data["date_stamp"] = price_model_data["date_stamp"].apply(lambda date: date.year)

# drop lines missing data used for prediction
price_model_data.dropna(subset=desired_cols, inplace=True)

# compute training data targets
price_targets = price_model_data["price"]
price_model_data.drop("price", axis=1, inplace=True)

# %%
# define the hyper-parameter's grid search
param_grid = {
    "max_depth" : np.linspace(13, 17, 5, dtype=int),
    "n_estimators" : np.linspace(100, 200, 11, dtype=int)
}

# Create the model
rfr = RandomForestRegressor(n_jobs=-1)
price_model = GridSearchCV(rfr, cv=2, param_grid=param_grid, n_jobs=-1, verbose=1)

# Train model
X_train, X_test, y_train, y_test = train_test_split(price_model_data, price_targets, test_size=0.2)
price_model.fit(X_train, y_train)
print(f"Train accuracy : {price_model.score(X_train, y_train):.2%}")
print(f"Test  accuracy : {price_model.score(X_test, y_test):.2%}")
print(f"Best params : {price_model.best_params_}")

# %%
# Compute the absolute price difference
predicted_price = price_model.predict(price_model_data)
predicted_price -= price_targets
predicted_price = np.abs(predicted_price)
predicted_price = pd.Series(predicted_price, index=price_model_data.index).sort_values(ascending=False)

# plot the first ten worst predictions
sns.barplot(x=merge["journal_name"][predicted_price[:10].index], y=predicted_price[:10])
plt.xticks(rotation=25, ha="right")
plt.show()

# %%
"""
Comme nous pouvons le constater sur le graphique ci-dessus, les 10 plus grandes erreurs de prédictions sont comprises
entre 2000 et 3000 en moyenne. Si on observe les données, nous pouvons nous rendre compte que les premières erreurs sont
souvent dues à des prédictions non nulles pour des journaux avec des prix nuls et inversement. Nous pouvons donc
conclure que notre modèle est suffisement précis pour prédire le prix moyen des journaux mais n'est pas encore capable
d'identifier des *outliers* qui proposent des prix bien différents des autres.
"""

# %%
"""
*Construire un modèle pour grouper les revues suivant le coût actuel de publication
(attribut « price ») et le score d’influence (attribut « proj_ai ») (cela inclut la
détermination du nombre de clusters, le choix et le paramétrage d’un modèle de
clustering, l’application du modèle pour trouver les clusters). Justifier les choix
effectués.*
"""

# %%
clustering_data = merge[["proj_ai", "price"]].dropna()
sns.scatterplot(x=clustering_data["proj_ai"], y=clustering_data["price"])

# %%
"""
D'après le graphique, il semble exister un grand cluster de journaux avec des prix compris entre 0 et 3000 et avec un
score d'influence entre 0 et 2. A coté de ce cluster, nous pouvons observer de nombreux points plus ou moins isolés qui
pourraient être considérés comme des *outliers*.

Cette disposition des points semble indiquer que l'approche par K-means pourrait avoir comme résultat de diviser le
grand cluster et de regrouper les outliers avec les sous clusters du grand cluster. Par conséquent, nous allons donc
nous interessé aux approches basées sur la densité. Nous allons aussi centrer et réduire les données.
"""

# %%
# Center and reduce data
clustering_model_data = StandardScaler().fit_transform(clustering_data)

# Create the models
clustering_models = [
    KMeans(n_clusters=3),
    AgglomerativeClustering(n_clusters=4, linkage="ward"),
    DBSCAN(eps=0.5, min_samples=5, n_jobs=-1),
    GaussianMixture(n_components=3, covariance_type="full")
]

# Run models, save and show results
predicted_clusters = []
for model in clustering_models:
    clusters = model.fit_predict(clustering_model_data)
    predicted_clusters.append(clusters)

    palette = sns.color_palette("hls", len(np.unique(clusters)))
    graph = sns.scatterplot(x=clustering_data["proj_ai"], y=clustering_data["price"], hue=clusters, palette=palette)
    graph.set_title(f"Clustering by {type(model).__name__}")
    plt.show()

# %%
"""
Comme nous l'avions prévu, certains clusterings dissocie le grand ensemble de points en différents clusters. Seul
`DBSCAN` semble bien identifier un grand cluster, quelques petits clusters et plusieurs *outliers*.
"""

# %%
"""
*Présenter des statistiques descriptives des clusters obtenus, et lister les revues du
meilleur cluster en termes de rapport moyen : score d’influence / coût de publication.*
"""

# %%

bestClusterIndex = None
bestClusterRatio = 0
for i, clusters in enumerate(predicted_clusters):
    clustering_data["clusters"] = clusters

    print(f"--- Statistiques du clustering de {type(clustering_models[i]).__name__} ---")
    for clusterId, cluster in clustering_data.groupby(["clusters"]):
        # skip outliers
        if clusterId == -1:
            continue

        print(f"\tCluster {clusterId} :")
        print(cluster[["price", "proj_ai"]].describe())

        mean_ratio = cluster['proj_ai'].mean() / (cluster['price'].mean() + 1e-3)
        print(f"Rapport moyen : {mean_ratio}")
        if mean_ratio > bestClusterRatio:
            bestClusterRatio = mean_ratio
            bestClusterIndex = cluster.index

        print()
    print()

# print the best journals according to its ration
print("Liste des meilleurs journaux selon leur rapport moyen :")
merge.iloc[bestClusterIndex][["journal_name", "pub_name"]]
