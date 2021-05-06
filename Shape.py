# -*- coding: utf-8 -*-
"""
Created on Mon May  3 17:09:20 2021

@author: 50604
"""

import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn import tree
import graphviz
import webbrowser
from sklearn.datasets import load_iris
# Set work direct
import os
os.getcwd()

os.chdir('D:\\MSSP\\Data\\679\\Final project\\SEER\\Python')
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin\\'


# Section 1, random forest model
data = pd.read_csv('../rfdata.csv')
X = data.iloc[:,2:23]
y = data.iloc[:,-1]
feature_names = [i for i in data.columns if data[i].dtype in [np.int64,np.float64]]
feature_names = feature_names[2:20]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

# Section 2, importance of features
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
html_obj=eli5.show_weights(perm, feature_names = val_X.columns.tolist())
with open('D:\\MSSP\\Data\\679\\Final project\\SEER\\Python\iris-importance.htm','wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored HTML file on the default browser
url = r'D:\\MSSP\\Data\\679\\Final project\\SEER\\Python\iris-importance.htm'
webbrowser.open(url, new=2)

# Section 3, decision tree model
y = data.iloc[:,-1]
feature_names = [i for i in data.columns if data[i].dtype in [np.int64,np.float64]]
feature_names = feature_names[2:20]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)

# Section 4, structure of decision tree
tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=feature_names)
graphviz.Source(tree_graph)


# Section 10, calculate SHAP value of a single prediction
row_to_show = 5
data_for_prediction = val_X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)

# Section 11, Visualization of SHAP
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction,matplotlib=True)

#html_obj=shap.initjs()
html_obj=shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
with open('D:\\MSSP\\Data\\679\\Final project\\SEER\\Python\iris-shap.htm','wb') as f:
    f.write(html_obj)

# Open the stored HTML file on the default browser
url = r'D:\\MSSP\\Data\\679\\Final project\\SEER\\Python\iris-shap.htm'
webbrowser.open(url, new=2)

# Section 14, summary plots
# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# calculate shap values. This is what we will plot.
# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
shap_values = explainer.shap_values(val_X)

# Make plot. Index of [1] is explained in text below.
shap.summary_plot(shap_values[1], val_X)



