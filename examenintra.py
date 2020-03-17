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
import re
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
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
- **is_hybrid** (valeur booléenne) : (D'après le site [FlourishOA](http://flourishoa.org/about#type)) Permet de savoir si le journal est
hybride. C'est-à-dire, si le journal est à abonnement avec certains articles en accès libre.
- **category** (valeur catégorique) : La liste des catégories de la revue scientifique.
- **url** (valeur catégorique) : L'adresse web de la page d'acceuil du journal
"""

# %%
#  Compute ratio of N/A values
(journal.isna().sum() / journal.shape[0]) * 100

# %%
"""
Nous pouvons remarquer que les colonnes `pub_name`, `category` et `url` possèdent des données manquantes. En particulier
`category` et `url` qui ont environs 50% de données manquantes.
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
"""
On peut voir qu'il existe de gros éditeurs, en particulier `Springer` qui publie 14.3% des revues scientifiques.
"""

# %%
journal["category"][11065:11070]

# %%
"""
Comme on peut le voir sur l'exemple ci-dessus, certaines revues possèdent plusieurs catégorie qui peuvent être séprarer
par plusieurs caractères comme `|`, `.` et `and`.

Finalement, pour la colonne `url`, il y a une grande partie de données manquantes mais 
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
`influence` ce qui consititue alors des valeurs abérantes.
- **url** (valeur catégorique) : L'adresse web de la revue vers la page d'informations pour les auteurs.
- **license** (valeur catégorique) : Valeur numérique représentant une lisence (nous n'avons pas d'information sur la
correspondance entre les valeurs numériques et les différentes lisences qui existent.
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
On remarque sur le graphique 4 groupes de prix des journaux. Un premier à 0, un deuxième entre 100 et 2900 environ, un
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
On note sur le graphique que la plupart des données ont été ajouté en 2016 et 2017, une autre partie fut ajouté en 2012
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
ont eu des mise à jours de leurs informations.
"""

# %%
# show ratio of incoherent values
print(f"Ratio de valeurs incohérentes pour influence_id : {1 - (price['influence_id'] <= influence.index[-1]).sum() / price.shape[0]:.2%}")

# %%
"""
La quasi-totalité des valeurs de `influence_id` sont soit manquantes soit ont une valeur supérieur à la valeur maximal
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
Bien qu'on ne retrouve qu'une seule valeur abérante, la majorité des valeurs reste manquente. Cela nuit donc fortement à
l'interêt de cette colonne.
"""

# %%
licenses, licenses_count = np.unique(price["license"].dropna(), return_counts=True)
plt.hist(licenses, bins=len(licenses), weights=licenses_count)
plt.show()

# %%
"""
Comme nous pouvons le voir sur l'histogramme, la lisence 2 est majoritairement utilisée, on retrouve ensuite les
lisences 10, 4 et 6. Cependant, ne pouvant faire l'equivalence entre ces numéros et le nom des licenses (ou groupes de
lisences), les informations de cette colonne ne sont pas pertinentes. 
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
- **citation_count_sum** (valeur continue) : Le nombre de citation du journal.
- **paper_count_sum** (valeur continue) : Le nombre d'articles scientifiques du journal.
- **avg_cites_per_paper** (valeur continue) : La moyenne du nombre de citation par article du journal.
- **proj_ai** (valeur continue) : Le score d'influence associé à la moyenne des citations.
- **proj_ai_year** (valeur temporelle) : La date associé au calcul du score d'influence.
"""

# %%
#  Compute ratio of N/A values
(influence.isna().sum() / influence.shape[0]) * 100

# %%
"""
Il n'y a presque aucune données manquantes dans la table `influence`. De plus, les quatres seules colonnes en possédant
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
On peut noter un très grande dispertion des valeurs du nombre de citations qui peut faire pense à une distribution de
Poisson. De nombreux journaux n'ont pas beaucoup de citation (environs 636) alors que certains journaux se dispersent
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
une dernière fois la Loi de Poisson avec un mode autour de la valeur 0.4 et une dispertion jusqu'à 11.
"""

# %%
print("Liste des valeurs uniques de proj_ai_year :")
for year in influence["proj_ai_year"].unique():
    print(f"\t- {year}")

# %%
"""
Surprenemment, nous pouvons noter que seule l'année 2015 est présente dans cette colonne.
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
nombre de valeur non manquantes + 5 si la colonne `category` est non manquante. Ce choix de privilégier la colonne
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
on peut trouver la mention de publiction fee de £330. On peut donc considérer la deuxième ligne avec un prix affiché de
$387.15 comme étant la bonne ligne.
"""

# %%
# Delete the wrong article found
price.drop(13073, inplace=True)

# Delete the other duplicates
price.drop_duplicates(subset=["date_stamp", "journal_id"], inplace=True)

# %%
"""
Nous avions aussi trouvé une URL incohérente lors de la question 1 et l'avons déjà supprimé à ce moment là.

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
faisant nous gardons les prix les plus récent afin d'obtenir des statistiques plus à jour.
"""

# %%
# merge tables with the desired columns
merge = journal.merge(price, left_on="issn", right_on="journal_id")

# sort by date stamp
merge.sort_values(by=["date_stamp"], ascending=False, inplace=True)

# only keep the first line of the duplicated journal name
merge.drop_duplicates(subset=["journal_name"], inplace=True)

# drop unwanted columns
merge.drop(merge.drop(["category", "price"], axis=1).columns, axis=1, inplace=True)

# drop row where either the price or the category is missing
merge.dropna(inplace=True)

merge = pd.concat([merge, merge["category"].str.get_dummies()], axis=1)

corr_cols = []
corr_vals = []
for col in merge.drop(["category", "price"], axis=1).columns:
    group = merge[merge[col] == 1]
    if group.shape[0] >= 10:
        corr_cols.append(col)
        corr_vals.append(group["price"].values)

sns.barplot(data=corr_vals)
plt.xticks(plt.xticks()[0], labels=corr_cols, rotation=55, ha="right")
plt.show()

# %%
"""
*Construire un modèle pour prédire les valeurs de catégorie de journaux manquantes de
la façon la plus précise possible (cela inclut la sélection d’attributs informatifs, le
choix et le paramétrage d’un modèle de classification, le calcul du score du modèle,
l’application du modèle pour prédire les catégories manquantes). Justifier les choix
effectués.*
"""

# %%


# %%
"""
## Question 3 : Régression-Clustering

*Supprimer tous les attributs ayant plus de 50% de données manquantes.*
"""

# %%


# %%
"""
*Construire un modèle pour prédire le coût actuel de publication (attribut « price ») à
partir des autres attributs (cela inclut la sélection d’attributs informatifs, le choix et le
paramétrage d’un modèle de régression, le calcul du score du modèle, l’application du
modèle pour prédire les coûts). Justifier les choix effectués.
Lister les 10 revues qui s’écartent le plus (en + ou -) de la valeur prédite.*
"""

# %%


# %%
"""
*Construire un modèle pour grouper les revues suivant le coût actuel de publication
(attribut « price ») et le score d’influence (attribut « proj_ai ») (cela inclut la
détermination du nombre de clusters, le choix et le paramétrage d’un modèle de
clustering, l’application du modèle pour trouver les clusters). Justifier les choix
effectués.*
"""

# %%


# %%
"""
*Présenter des statistiques descriptives des clusters obtenus, et lister les revues du
meilleur cluster en termes de rapport moyen : score d’influence / coût de publication.*
"""

# %%


