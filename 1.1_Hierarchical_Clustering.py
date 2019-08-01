#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 17:47:50 2019

@author: jaime.iglesias

Purpose: To practice clustering on the wholesale customers dataset
"""

"""
    This dataset is a collection of the annual spending of wholesale customers
    of a company based in Portugal. The monetary units are unknown, and the
    categorical information related to each client is as follows:
        
        Channel = {1 : 'Horeca',
                   2 : 'Retail'}
        
        Region = {1 : 'Lisbon',
                  2 : 'Oporto',
                  3 : 'Other Region',}
"""


# Importing new libraries
from scipy.cluster.hierarchy import dendrogram, linkage # dendrograms
from sklearn.cluster import KMeans # k-means clustering


# Importing known libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)



# Importing dataset
customers_df = pd.read_excel('dataset_wholesale_customers.xlsx')



"""
    Do we need to scale for hierarchical clustering? Technically, no. However,
    in practice this depends on our data. Let's keep this in mind as we walk
    through our exploratory data analysis.
"""




###############################################################################
# Agglomerative Clustering
###############################################################################


"""
    Agglomerative clustering starts where each observation is its own cluster.
    For there, it links observations together based on distance. There are
    three primary methods to calculate distance.
    
    ward (default) - groups observations into clusters in a way that minimizes 
    the variance amongst all clusters. Leads to clusters that are relatively
    equal in size.

    average - merges clusters that have the smallest average distance
    between all their points.

    complete - merges clusters that have the smallest maximum distance
    between their points.
"""



"""
    A major ADVANTAGE of agglomerative clustering is the dendrogram
    A major DRAWBACK of agglomerative clustering is that it is unable to
    predict on new data.
    
"""



###############################################################################
# Scaling Before Agglomerative Clustering
###############################################################################


########################
# Scaling WITHOUT demographic variables
########################


# Scaling using StandardScaler()
scaler = StandardScaler()


scaler.fit(customers_df.iloc[ : , 2:])


X_scaled = scaler.transform(customers_df.iloc[ : , 2:])



# Concatinating with categorical data
X_scaled_df = pd.DataFrame(X_scaled)



# Renaming columns
X_scaled_df.columns = customers_df.iloc[ : , 2:].columns



###############################################################################
# Building a Dendrogram
###############################################################################

standard_mergings_ward = linkage(y = X_scaled_df,
                                 method = 'ward')


fig, ax = plt.subplots(figsize=(8, 8))

dendrogram(Z = standard_mergings_ward,
           leaf_rotation = 90,
           leaf_font_size = 6)


plt.savefig('standard_hierarchical_clust_ward.png')
plt.show()


###############################################################################
# Hierarchical Clustering: K-Means
###############################################################################

"""
    If we know how many clusters we would like to observe, we can take
    advantage of k-means clustering. This is a nice way to divide data into
    clusters to gain further insights.
"""


# Creating a model with k clusters
customers_k3 = KMeans(n_clusters = 3,
                      random_state = 508)



# Fit model to points
customers_k3.fit(X_scaled_df)



# Checking to see if we got the same clusters as when using fcluster 
customers_kmeans_clusters = pd.DataFrame({'cluster': customers_k3.labels_})



########################
# How much data is in each cluster?
########################

print(customers_kmeans_clusters.iloc[: , 0].value_counts())





########################
# What are the centriods (variable averages) for each cluster?
########################

centroids = customers_k3.cluster_centers_



centroids_df = pd.DataFrame(centroids)



# Renaming columns
centroids_df.columns = customers_df.iloc[ : , 2:].columns



# Sending data to Excel
centroids_df.to_excel('customers_k3_centriods.xlsx')



"""
    We can use the above DataFrame to understand the characteristics of each
    cluster. This way we interpret the potential segments in our data.
"""




###############################################################################
# Plotting Intertia
###############################################################################

"""
    How many clusters do we need? Which number of clusters is 'best' for our
    data? These questions can be answered using the metric inertia.
"""


ks = range(1, 50)
inertias = []


for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters = k)


    # Fit model to samples
    model.fit(X_scaled_df)


    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)



# Plot ks vs inertias
fig, ax = plt.subplots(figsize = (12, 8))
plt.plot(ks, inertias, '-o')


plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)


plt.show()



########################
# KMeans with 9 clusters
########################


# Creating a model with 9 clusters
customers_k9 = KMeans(n_clusters = 9,
                      random_state = 508)



# Fit model to points (removing channel and region)
customers_k9.fit(X_scaled_df)



# Checking to see if we got the same clusters as when using fcluster 
customers_k9_clusters = pd.DataFrame({'cluster': customers_k9.labels_})



print(customers_k9_clusters.iloc[: , 0].value_counts())



centroids_k9 = customers_k9.cluster_centers_



centroids_k9_df = pd.DataFrame(centroids_k9)



centroids_k9_df.columns = customers_df.columns[2:len(customers_df.columns)]
centroids_k9_df.to_excel('cutomers_k9_centriods.xlsx')



"""

    A great aspect of KMeans clustering is that it can predict on new
    observations.
"""


###############################################################################
# Advanced: Hierarchical Clustering: Predicting with K-Means
###############################################################################


########################
# 3 Clusters - Predicting members of each cluster
########################

# Pulling in our cluster labels from when we made three clusters
print(customers_k3.labels_)


# Train/test split - Region
X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df,
        customers_k3.labels_,
        test_size = 0.25,
        random_state = 508)



c_k3 = KMeans(n_clusters = 3,
                     random_state = 508)



# Fit model to points
c_k3.fit(X_train, y_train)



# Predicting on new data
c_k3_pred = c_k3.predict(X_test)



# Creating a DataFrame for to analyze results using crosstab
h_clust_pred_df = pd.DataFrame({'Actual' : y_test,
                                'Predicted': c_k3_pred})


    
h_clust_crosstab = pd.crosstab(h_clust_pred_df['Actual'],
                               h_clust_pred_df['Predicted'])


print(h_clust_crosstab)




########################
# 9 Clusters - Predicting members of each cluster
########################

# Pulling in our cluster labels from when we made three clusters
print(customers_k9.labels_)


# Train/test split - Region
X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df,
        customers_k9.labels_,
        test_size = 0.25,
        random_state = 508)



c_k9 = KMeans(n_clusters = 9,
              random_state = 508)



# Fit model to points
c_k9.fit(X_train, y_train)



# Predicting on new data
c_k9_pred = c_k9.predict(X_test)



# Creating a DataFrame for to analyze results using crosstab
h_clust_pred_df = pd.DataFrame({'Actual' : y_test,
                                'Predicted': c_k9_pred})


    
h_clust_crosstab = pd.crosstab(h_clust_pred_df['Actual'],
                               h_clust_pred_df['Predicted'])


print(h_clust_crosstab)



###############################################################################
# Advanced: Predicting Region and Channel
###############################################################################

########################
# Creating a model with 3 clusters, one for each region
########################

# Train/test split - Region
X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df,
        customers_df.iloc[ : , 1 ],
        test_size = 0.25,
        random_state = 508)




c_region_k3 = KMeans(n_clusters = 5,
                     random_state = 508)



# Fit model to points
c_region_k3.fit(X_train, y_train)



# Predicting on new data
c_region_k3_pred = c_region_k3.predict(X_test)



# Creating a DataFrame for to analyze results using crosstab
h_clust_pred_df = pd.DataFrame({'Actual' : y_test,
                                'Predicted': c_region_k3_pred})


    
h_clust_crosstab = pd.crosstab(h_clust_pred_df['Actual'],
                               h_clust_pred_df['Predicted'])


print(h_clust_crosstab)



########################
# Creating a model with 9 clusters
########################

c_region_k9 = KMeans(n_clusters = 9,
                     random_state = 508)



# Fit model to points
c_region_k9.fit(X_train, y_train)



# Predicting on new data
c_region_k9_pred = c_region_k9.predict(X_test)



"""
    Notice that we cannot use the score method to check the accuracy of our
    predictions. This is because the clusters are very likely to have a
    different numbering when compared to our regions.
"""


h_clust_pred_df = pd.DataFrame({'Actual' : y_test,
                                'Predicted': c_region_k9_pred})


    
h_clust_crosstab = pd.crosstab(h_clust_pred_df['Actual'],
                               h_clust_pred_df['Predicted'])


print(h_clust_crosstab)



"""
    It doesn't appear that our regions don't separate well based on our
    clusters. Let's analyze our results when predicting channel.
"""



########################
# Predicting channel
########################

# Train/test split - Channel
X_train, X_test, y_train, y_test = train_test_split(
        customers_df.iloc[ : , 2: ],
        customers_df.iloc[ : , 0 ],
        test_size = 0.25,
        random_state = 508)



########################
# Creating a model with 2 clusters, one for each channel
########################

c_channel_k2 = KMeans(n_clusters = 2,
                     random_state = 508)



c_channel_k2.fit(X_train, y_train)
c_channel_k2_pred = c_channel_k2.predict(X_test)


# Analyzing results
h_clust_pred_df = pd.DataFrame({'Actual' : y_test,
                                'Predicted': c_channel_k2_pred})



h_clust_crosstab = pd.crosstab(h_clust_pred_df['Actual'],
                               h_clust_pred_df['Predicted'])


print(h_clust_crosstab)



########################
# Creating a model with 9 clusters
########################

c_channel_k9 = KMeans(n_clusters = 9,
                      random_state = 508)



# Fit model to points
c_channel_k9.fit(X_train, y_train)



# Predicting on new data
c_channel_k9_pred = c_channel_k9.predict(X_test)



"""
    Notice that we cannot use the score method to check the accuracy of our
    predictions. This is because the clusters are very likely to have a
    different numbering when compared to our regions.
"""


h_clust_pred_df = pd.DataFrame({'Actual' : y_test,
                                'Predicted': c_channel_k9_pred})


    
h_clust_crosstab = pd.crosstab(h_clust_pred_df['Actual'],
                               h_clust_pred_df['Predicted'])


print(h_clust_crosstab)



"""
    With 9 clusters, we are seeing a reasonable degree of separation between
    the two channels. Since we can consider each channel as its own 'pseudo-
    cluster', we have increase the number of clusters we have from 9 to 18.
    
    Let's check the centroids of our clusters so that we may learn more about
    our segments.
"""








