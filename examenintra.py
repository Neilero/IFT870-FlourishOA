# %%
"""
# IFT870 - Examen intratrimestriel

Auteur : Aurélien Vauthier (19 126 456)
"""

# %%

# import libraries
import numpy as np
import pandas as pd
import re
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

# extract data
journal = pd.read_csv("api_journal11-13-17.csv", encoding="windows-1252")
price = pd.read_csv("api_price11-13-17.csv", index_col=0)
influence = pd.read_csv("estimated-article-influence-scores-2015.csv", index_col=0)

# %%
"""
## Question 1 : Exploration-Description

*Présenter une description de chacun des attributs des 3 tables, avec des graphiques
pour la visualisation des statistiques descriptives au besoin.*
"""

# %%

journal.head()

# %%
"""
## Question 2 : Prétraitement-Représentation

*Effectuer un prétraitement des données pour supprimer les duplications et corriger les
incohérences s’il y en a.*
"""

# %%


# %%
"""
*Y a t il une corrélation entre les catégories de journaux (attribut « category ») et les
coûts de publication (attribut « price ») ? Justifier la réponse.*
"""

# %%


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


