#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 17:47:50 2019

@author: jaime.iglesias

Purpose: To practice clustering on the wholesale customers dataset
"""

"""
Prof. Chase:
    This dataset is a collection of the annual spending of wholesale customers
    of a company based in Portugal. The monetary units are unknown, and the
    categorical information related to each client is as follows:
        
        Channel = {1 : 'Horeca',
                   2 : 'Retail'}
        
        Region = {1 : 'Lisbon',
                  2 : 'Oporto',
                  3 : 'Other Region'}
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
# Exploratory Data Analysis
###############################################################################

customers_df.info()


customers_desc = customers_df.describe(percentiles = [0.01,
                                                      0.05,
                                                      0.10,
                                                      0.25,
                                                      0.50,
                                                      0.75,
                                                      0.90,
                                                      0.95,
                                                      0.99]).round(2)



customers_desc.loc[['min',
                    '1%',
                    '5%',
                    '10%',
                    '25%',
                    'mean',
                    '50%',
                    '75%',
                    '90%',
                    '95%',
                    '99%',
                    'max'], :]




# Checking class balances
print(customers_df['Channel'].value_counts())
print(customers_df['Region'].value_counts())




# Viewing the first few rows of the data
customers_df.head(n = 5)





"""
    * 8 variables of type int64
    * no missing values
    * no negative values
    * many distributions seem skewed
"""



########################
# Histograms
########################

# Plotting categorical information
fig, ax = plt.subplots(figsize=(12,8))
plt.subplot(2, 1, 1)
sns.distplot(a = customers_df['Channel'],
             hist = True,
             kde = True,
             color = 'blue')



plt.subplot(2, 1, 2)
sns.distplot(a = customers_df['Region'],
             hist = True,
             kde = True,
             color = 'red')


plt.show()


# Plotting numeric information

fig, ax = plt.subplots(figsize = (12, 8))


plt.subplot(2, 3, 1)
sns.distplot(a = customers_df['Fresh'],
             hist = True,
             kde = True,
             color = 'purple')



plt.subplot(2, 3, 2)
sns.distplot(a = customers_df['Milk'],
             hist = True,
             kde = True,
             color = 'green')



plt.subplot(2, 3, 3)
sns.distplot(a = customers_df['Grocery'],
             hist = True,
             kde = True,
             color = 'yellow')



plt.subplot(2, 3, 4)
sns.distplot(a = customers_df['Frozen'],
             hist = True,
             kde = True,
             color = 'red')



plt.subplot(2, 3, 5)
sns.distplot(a = customers_df['Detergents_Paper'],
             hist = True,
             kde = True,
             color = 'orange')



plt.subplot(2, 3, 6)
sns.distplot(a = customers_df['Delicassen'],
             hist = True,
             kde = True,
             color = 'pink')



plt.tight_layout()
plt.savefig('wholesale_raw_plots.png')
plt.show()



########################
# Correlation analysis
########################

fig, ax = plt.subplots(figsize = (8, 8))


df_corr = customers_df.corr().round(2)


sns.heatmap(df_corr,
            cmap = 'coolwarm',
            square = True,
            annot = True)


plt.savefig('wholesale_correlations.png')
plt.show()



"""
    Even though there is not a lot of correlation amongst the variables, let's
    see how much variance can be explained through principal components.
"""



########################
# Scaling (normalizing) variables before correlation analysis
########################


"""
    Note that normally we should avoid scaling categorical information in the
    way that we are about to, even if it's coded as numeric. However, this
    example is meant to show the effects standardized scaling has on
    Pearson correlation.
"""

# Scaling the wholesale customer dataset
customer_features = customers_df.iloc[ : , : ]



# Scaling using StandardScaler()
scaler = StandardScaler()



scaler.fit(customer_features)



X_scaled = scaler.transform(customer_features)



X_scaled_df = pd.DataFrame(X_scaled)




#  Checking pre- and post-scaling of the data
print(pd.np.var(customer_features))
print(pd.np.var(X_scaled_df))



# Building a heatmap of the scaled correlations
fig, ax = plt.subplots(figsize = (8, 8))


df_scaled_corr = X_scaled_df.corr().round(2)


sns.heatmap(df_scaled_corr,
            cmap = 'coolwarm',
            square = True,
            annot = True)


plt.savefig('standardized_wholesale_correlations.png')
plt.show()



# Comparing the two heatmaps

fig, ax = plt.subplots(figsize = (8, 8))


plt.subplot(1, 2, 1)
sns.heatmap(df_corr, cmap = 'coolwarm', square = True, annot = True, cbar = False)


plt.subplot(1, 2, 2)
sns.heatmap(df_scaled_corr, cmap = 'coolwarm', square = True, annot = True, cbar = False)


plt.show()




###############################################################################
# Principal Component Analysis (PCA)
###############################################################################


"""
    Principal component analysis is primarily conducted for three reasons:
        1) A solution for when explanatory variables are correlated
           (leading to multicollinearity), which is a violation of one of the
           assumptions of linear models.
        
        2) Dimensionality reduction, which allows for modeling when there are
           too many explanatory variables
           
        3) Latent trait exploration (i.e. understanding ones leadership ability
           (cannot be directly measured) through measurabe explanatory
           variables.
"""



########################
# Performing PCA on the scaled data
########################


# Looking at all of the principal components
customer_pca = PCA(n_components = None,
                   random_state = 508)



# Fitting the PCA model
customer_pca.fit(X_scaled)



# Transform data
customer_pca.transform(X_scaled)


print("Original shape:", X_scaled.shape)
print("Reduced shape:",  customer_pca.transform(X_scaled).shape)



# Explained variance as a ratio of total variance
customer_pca.explained_variance_ratio_



# Plotting the principal components
fig, ax = plt.subplots(figsize=(10, 8))

features = range(customer_pca.n_components_)


plt.bar(x = features,
        height = customer_pca.explained_variance_ratio_)


plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)

plt.show()



"""
    Bar plots are a good way to understand the variance explained by principal
    components. Traditionally, however, we use a scree plot for this.
"""



###############################################################################
# Building a Scree Plot
###############################################################################

fig, ax = plt.subplots(figsize=(10, 8))

features = range(customer_pca.n_components_)


plt.plot(features,
         customer_pca.explained_variance_ratio_,
         linewidth = 2,
         marker = 'o',
         markersize = 10,
         markeredgecolor = 'black',
         markerfacecolor = 'grey')


plt.title('Wholesale Customer Scree Plot')
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)

plt.savefig('wholesale_customer_pca_scree_plot.png')
plt.show()


########################
# Cumulative variance
########################

print(f"""

Normally, we may want to set a threshold at explaining 80% of the variance
in the dataset.

Right now we have the following:
    1 Principal Component : {customer_pca.explained_variance_ratio_[0].round(2)}
    2 Principal Components: {(customer_pca.explained_variance_ratio_[0] + customer_pca.explained_variance_ratio_[1]).round(2)}
    3 Principal Components: {(customer_pca.explained_variance_ratio_[0] + customer_pca.explained_variance_ratio_[1] + customer_pca.explained_variance_ratio_[2]).round(2)}
    4 Principal Components: {(customer_pca.explained_variance_ratio_[0] + customer_pca.explained_variance_ratio_[1] + customer_pca.explained_variance_ratio_[2] + customer_pca.explained_variance_ratio_[3]).round(2)}

""")




###############################################################################
# Re-evaluating The Scaling
###############################################################################

"""
    Let's try running PCA again without the categorical data. One leading
    school of thought when conducting segmentation based on customer behaviors
    is, well, behavior data. This implies that data such as region and channel
    should be removed from this part of the analysis.
"""



# Subsetting the wholesale customer dataset
customer_features_reduced = customers_df.iloc[ : , 2: ]



# Scaling steps
scaler = StandardScaler()



scaler.fit(customer_features_reduced)



X_scaled_reduced = scaler.transform(customer_features_reduced)


# PCA steps
customer_pca_reduced = PCA(n_components = None,
                           random_state = 508)



customer_pca_reduced.fit(X_scaled_reduced)



X_pca_reduced = customer_pca_reduced.transform(X_scaled_reduced)



print("Original shape:", X_scaled_reduced.shape)
print("Reduced shape:",  X_pca_reduced.shape)




# Explained variance as a ratio of total variance
customer_pca_reduced.explained_variance_ratio_




# Building a scree plot
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

plt.savefig('reduced_wholesale_customer_pca_scree_plot.png')
plt.show()



print(f"""

Normally, we may want to set a threshold at explaining 80% of the variance
in the dataset.

Right now we have the following:
    1 Principal Component : {customer_pca_reduced.explained_variance_ratio_[0].round(2)}
    2 Principal Components: {(customer_pca_reduced.explained_variance_ratio_[0] + customer_pca.explained_variance_ratio_[1]).round(2)}
    3 Principal Components: {(customer_pca_reduced.explained_variance_ratio_[0] + customer_pca.explained_variance_ratio_[1] + customer_pca.explained_variance_ratio_[2]).round(2)}
    4 Principal Components: {(customer_pca_reduced.explained_variance_ratio_[0] + customer_pca.explained_variance_ratio_[1] + customer_pca.explained_variance_ratio_[2] + customer_pca.explained_variance_ratio_[3]).round(2)}

""")


    
    
###############################################################################
# Factor Loadings (What's Inside of our Principal Components)
###############################################################################


"""
    Since the variance amongst factors is getting "bundled" into principal
    components, it is a good idea to try and interpret each component.
"""


# Plotting the principal components
plt.matshow(customer_pca_reduced.components_, 
            cmap = 'Blues')



plt.yticks([0, 1, 2, 3, 4, 5],
           ["PC 1", "PC 2", "PC 3", "PC 4", "PC 5", "PC 6"])


plt.colorbar()


plt.xticks(range(0, 6),
           customers_df.columns[2:],
           rotation=60,
           ha='left')


plt.xlabel("Feature")
plt.ylabel("Principal components")



########################
# Checking factor loadings
########################


factor_loadings_df = pd.DataFrame(pd.np.transpose(customer_pca_reduced.components_))


factor_loadings_df = factor_loadings_df.set_index(customers_df.columns[2:])


print(factor_loadings_df)


factor_loadings_df.to_excel('customer_factor_loadings.xlsx')







###############################################################################
###############################################################################
# Optional: Not Required for the Final
###############################################################################
###############################################################################

###############################################################################
# PCA Predicting with PCA
###############################################################################

"""
    The factor loadings of PCA can used as explanatory factors for regression
    or classification. This, of course, comes with the caveat that we have a
    response variable.
    
    Let's try predicting channel based on our factor loadings.
"""


# Importing logistic regression library and train/test split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



# Partioning the response variable
target_channel = customers_df.loc[ : , 'Channel']


########################
# Predicting channel with PCA/logisitic regression
########################

# Train/Test Split - Channel
X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_reduced,
            target_channel,
            test_size = 0.20,
            random_state = 508)




# Applying PCA with three factors
pca_3 =       PCA(n_components = 3,
              random_state = 508)



X_train_pca = pca_3.fit_transform(X_train)

X_test_pca  = pca_3.fit_transform(X_test)



# Testing the lengths of the new objects
len(X_train_pca) == len(X_train) and len(X_test_pca) == len(X_test)



# Checking the test observations
pd.DataFrame(X_test_pca)




# Applying PCA classification
log_reg_pca = LogisticRegression(solver = 'lbfgs')


"""
    Setting the slover to lbfgs will speed up processing. We will use this
    solver method here as well as when validating the quality of our model
    using KNN.
"""


log_reg_pca.fit(X_train, y_train)



y_pred_pca = log_reg_pca.predict(X_test)



# Scoring the model
y_score_train_pca = log_reg_pca.score(X_train, y_train)
y_score_test_pca = log_reg_pca.score(X_test, y_test)


print(y_score_train_pca)
print(y_score_test_pca)



########################
# Predicting channel with standard logisitic regression
########################

# Creating a new train/test split
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
            customer_features_reduced,
            target_channel,
            test_size = 0.20,
            random_state = 508)




# Applying PCA classification
log_reg = LogisticRegression(solver = 'lbfgs')



log_reg.fit(X_train_2, y_train_2)



y_pred_lg = log_reg.predict(X_test_2)



# Scoring the model
y_score_lg = log_reg.score(X_test_2, y_test_2)

print(y_score_lg)

# Scoring the model
y_score_train_lg = log_reg.score(X_train_2, y_train_2)
y_score_test_lg = log_reg.score(X_test_2, y_test_2)


print(y_score_train_lg)
print(y_score_train_lg)



###############################################################################
# Which model predicts better?
###############################################################################

print(f"""
PCA Predicted Train : {y_score_train_pca.round(3)}
PCA Predicted Test  : {y_score_test_pca.round(3)}
                             
LR Predicted Train  : {y_score_train_lg.round(3)}
LR Predicted Test   : {y_score_test_lg.round(3)}
""")


"""
    Remember, when we have X variables that are highly correlated and/or we
     have a large number of X variables, PCA can be a great option.
"""


    
