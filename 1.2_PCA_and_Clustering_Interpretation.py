#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 21:10:50 2019

@author: jaime.iglesias
"""


# Importing new libraries
from sklearn.preprocessing import StandardScaler # standard scaler
from sklearn.decomposition import PCA # principal component analysis


# Importing known libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)



# Importing dataset
customers_df = pd.read_excel('dataset_wholesale_customers.xlsx')



###############################################################################
# PCA One More Time!!!
###############################################################################



########################
# Step 1: Remove demographic information
########################

customer_features_reduced = customers_df.iloc[ : , 2: ]



########################
# Step 2: Scale to get equal variance
########################

scaler = StandardScaler()


scaler.fit(customer_features_reduced)


X_scaled_reduced = scaler.transform(customer_features_reduced)



########################
# Step 3: Run PCA without limiting the number of components
########################

customer_pca_reduced = PCA(n_components = None,
                           random_state = 508)


customer_pca_reduced.fit(X_scaled_reduced)


X_pca_reduced = customer_pca_reduced.transform(X_scaled_reduced)



########################
# Step 4: Analyze the scree plot to determine how many components to retain
########################

fig, ax = plt.subplots(figsize=(10, 8))

features = range(customer_pca_reduced.n_components_)


plt.plot(features,
         customer_pca_reduced.explained_variance_ratio_,
         linewidth = 2,
         marker = 'o',
         markersize = 10,
         markeredgecolor = 'black',
         markerfacecolor = 'grey')


plt.title('Reduced Wholesale Customer Scree Plot')
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)
plt.show()




########################
# Step 5: Run PCA again based on the desired number of components
########################

customer_pca_reduced = PCA(n_components = 3,
                           random_state = 508)


customer_pca_reduced.fit(X_scaled_reduced)



########################
# Step 6: Analyze factor loadings to understand principal components
########################

factor_loadings_df = pd.DataFrame(pd.np.transpose(customer_pca_reduced.components_))


factor_loadings_df = factor_loadings_df.set_index(customers_df.columns[2:])


print(factor_loadings_df)


factor_loadings_df.to_excel('practice_factor_loadings.xlsx')



########################
# Step 7: Analyze factor strengths per customer
########################

X_pca_reduced = customer_pca_reduced.transform(X_scaled_reduced)


X_pca_df = pd.DataFrame(X_pca_reduced)



########################
# Step 8: Rename your principal components and reattach demographic information
########################

X_pca_df.columns = ['Coffee_Shop_Essentials', 'Food_Items', 'Artistic_Pairings']


final_pca_df = pd.concat([customers_df.loc[ : , ['Channel', 'Region']] , X_pca_df], axis = 1)




########################
# Step 9: Analyze in more detail
########################


# Renaming channels
channel_names = {1 : 'Horeca',
                 2 : 'Retail'}


final_pca_df['Channel'].replace(channel_names, inplace = True)



# Renaming regions
region_names = {1 : 'Lisbon',
                2 : 'Oporto',
                3 : 'Other Region'}


final_pca_df['Region'].replace(region_names, inplace = True)



# Analyzing by channel
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Channel',
            y =  'Coffee_Shop_Essentials',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Channel',
            y =  'Food_Items',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Channel',
            y =  'Artistic_Pairings',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()




# Analyzing by region
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Region',
            y =  'Coffee_Shop_Essentials',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Region',
            y =  'Food_Items',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Region',
            y =  'Artistic_Pairings',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()





###############################################################################
# Cluster Analysis One More Time!!!
###############################################################################

from sklearn.cluster import KMeans # k-means clustering


########################
# Step 1: Remove demographic information
########################

customer_features_reduced = customers_df.iloc[ : , 2: ]



########################
# Step 2: Scale to get equal variance
########################

scaler = StandardScaler()


scaler.fit(customer_features_reduced)


X_scaled_reduced = scaler.transform(customer_features_reduced)



########################
# Step 3: Experiment with different numbers of clusters
########################

customers_k = KMeans(n_clusters = 5,
                      random_state = 508)


customers_k.fit(X_scaled_reduced)


customers_kmeans_clusters = pd.DataFrame({'cluster': customers_k.labels_})


print(customers_kmeans_clusters.iloc[: , 0].value_counts())



########################
# Step 4: Analyze cluster centers
########################

centroids = customers_k.cluster_centers_


centroids_df = pd.DataFrame(centroids)



# Renaming columns
centroids_df.columns = customer_features_reduced.columns


print(centroids_df)


# Sending data to Excel
centroids_df.to_excel('customers_k3_centriods.xlsx')



########################
# Step 5: Analyze cluster memberships
########################


X_scaled_reduced_df = pd.DataFrame(X_scaled_reduced)


X_scaled_reduced_df.columns = customer_features_reduced.columns


clusters_df = pd.concat([customers_kmeans_clusters,
                         X_scaled_reduced_df],
                         axis = 1)


print(clusters_df)



########################
# Step 6: Reattach demographic information 
########################

final_clusters_df = pd.concat([customers_df.loc[ : , ['Channel', 'Region'] ],
                               clusters_df],
                               axis = 1)


print(final_clusters_df)



########################
# Step 7: Analyze in more detail 
########################

# Renaming channels
channel_names = {1 : 'Horeca',
                 2 : 'Retail'}


final_clusters_df['Channel'].replace(channel_names, inplace = True)



# Renaming regions
region_names = {1 : 'Lisbon',
                2 : 'Oporto',
                3 : 'Other Region'}


final_clusters_df['Region'].replace(region_names, inplace = True)



########################
# Channel
########################

# Fresh
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Channel',
            y = 'Fresh',
            hue = 'cluster',
            data = final_clusters_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



# Milk
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Channel',
            y = 'Milk',
            hue = 'cluster',
            data = final_clusters_df)

plt.ylim(-2, 6)
plt.tight_layout()
plt.show()



# Grocery
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Channel',
            y = 'Grocery',
            hue = 'cluster',
            data = final_clusters_df)

plt.ylim(-2, 6)
plt.tight_layout()
plt.show()



# Frozen
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Channel',
            y = 'Frozen',
            hue = 'cluster',
            data = final_clusters_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



# Detergents_Paper
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Channel',
            y = 'Detergents_Paper',
            hue = 'cluster',
            data = final_clusters_df)

plt.ylim(-2, 5)
plt.tight_layout()
plt.show()



# Delicassen
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Channel',
            y = 'Delicassen',
            hue = 'cluster',
            data = final_clusters_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



########################
# Region
########################

# Fresh
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Region',
            y = 'Fresh',
            hue = 'cluster',
            data = final_clusters_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



# Milk
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Region',
            y = 'Milk',
            hue = 'cluster',
            data = final_clusters_df)

plt.ylim(-2, 6)
plt.tight_layout()
plt.show()



# Grocery
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Region',
            y = 'Grocery',
            hue = 'cluster',
            data = final_clusters_df)

plt.ylim(-2, 6)
plt.tight_layout()
plt.show()



# Frozen
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Region',
            y = 'Frozen',
            hue = 'cluster',
            data = final_clusters_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



# Detergents_Paper
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Region',
            y = 'Detergents_Paper',
            hue = 'cluster',
            data = final_clusters_df)

plt.ylim(-2, 5)
plt.tight_layout()
plt.show()



# Delicassen
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Region',
            y = 'Delicassen',
            hue = 'cluster',
            data = final_clusters_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



###############################################################################
# Combining PCA and Clustering!!!
###############################################################################

"""
Prof. Chase:
    That's right! We can combine both techniques! 
"""

########################
# Step 1: Take your transformed dataframe
########################

print(X_pca_df.head(n = 5))


print(pd.np.var(X_pca_df))



########################
# Step 2: Scale to get equal variance
########################

scaler = StandardScaler()


scaler.fit(X_pca_df)


X_pca_clust = scaler.transform(X_pca_df)


X_pca_clust_df = pd.DataFrame(X_pca_clust)


print(pd.np.var(X_pca_clust_df))


X_pca_clust_df.columns = X_pca_df.columns



########################
# Step 3: Experiment with different numbers of clusters
########################

customers_k_pca = KMeans(n_clusters = 5,
                         random_state = 508)


customers_k_pca.fit(X_pca_clust_df)


customers_kmeans_pca = pd.DataFrame({'cluster': customers_k_pca.labels_})


print(customers_kmeans_pca.iloc[: , 0].value_counts())




########################
# Step 4: Analyze cluster centers
########################

centroids_pca = customers_k_pca.cluster_centers_


centroids_pca_df = pd.DataFrame(centroids_pca)


# Rename your principal components
centroids_pca_df.columns = ['Coffee_Shop_Essentials', 'Food_Items', 'Artistic_Pairings']


print(centroids_pca_df)


# Sending data to Excel
centroids_pca_df.to_excel('customers_pca_centriods.xlsx')



########################
# Step 5: Analyze cluster memberships
########################

clst_pca_df = pd.concat([customers_kmeans_pca,
                         X_pca_clust_df],
                         axis = 1)


print(clst_pca_df)



########################
# Step 6: Reattach demographic information
########################

final_pca_clust_df = pd.concat([customers_df.loc[ : , ['Channel', 'Region']],
                                clst_pca_df],
                                axis = 1)


print(final_pca_clust_df.head(n = 5))



########################
# Step 7: Analyze in more detail 
########################

# Renaming channels
channel_names = {1 : 'Horeca',
                 2 : 'Retail'}


final_pca_clust_df['Channel'].replace(channel_names, inplace = True)



# Renaming regions
region_names = {1 : 'Lisbon',
                2 : 'Oporto',
                3 : 'Other Region'}


final_pca_clust_df['Region'].replace(region_names, inplace = True)


# Adding a productivity step
data_df = final_pca_clust_df



########################
# Channel
########################

# Coffee_Shop_Essentials
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Channel',
            y = 'Coffee_Shop_Essentials',
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()



# Food_Items
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Channel',
            y = 'Food_Items',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 6)
plt.tight_layout()
plt.show()



# Artistic_Pairings
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Channel',
            y = 'Artistic_Pairings',
            hue = 'cluster',
            data = data_df)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()



########################
# Region
########################

# Coffee_Shop_Essentials
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Region',
            y = 'Coffee_Shop_Essentials',
            hue = 'cluster',
            data = data_df)

plt.ylim(-1, 8)
plt.tight_layout()
plt.show()



# Food_Items
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Region',
            y = 'Food_Items',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 5)
plt.tight_layout()
plt.show()



# Artistic_Pairings
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Region',
            y = 'Artistic_Pairings',
            hue = 'cluster',
            data = data_df)

plt.ylim(-4, 3)
plt.tight_layout()
plt.show()



