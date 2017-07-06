from __future__ import print_function
import os
import subprocess
import math
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn import datasets
#import encode_target
#from encode_target import encode_target
#import visualize_tree
#from visualize_tree import visualize_tree
#import get_code
#from get_code import get_code


#####################################################
## auxilliary function
def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    #df_mod[target_column] = df_mod[target_column].replace(map_to_int)
    df_mod[target_column]=df_mod[target_column].replace(map_to_int)

    return (df_mod ,targets)
    
    
    
def get_code(tree, feature_names, target_names,
             spacer_base="    "):
    """Produce psuedo-code for decision tree.

    Args
    ----
    tree -- scikit-leant DescisionTree.
    feature_names -- list of feature names.
    target_names -- list of target (class) names.
    spacer_base -- used for spacing code (default: "    ").

    Notes
    -----
    based on http://stackoverflow.com/a/30104792.
    """
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node, depth):
        spacer = spacer_base * depth
        if (threshold[node] != -2):
            print(spacer + "if ( " + features[node] + " <= " + \
                  str(threshold[node]) + " ) {")
            if left[node] != -1:
                    recurse(left, right, threshold, features,
                            left[node], depth+1)
            print(spacer + "}\n" + spacer +"else {")
            if right[node] != -1:
                    recurse(left, right, threshold, features,
                            right[node], depth+1)
            print(spacer + "}")
        else:
            target = value[node]
            for i, v in zip(np.nonzero(target)[1],
                            target[np.nonzero(target)]):
                target_name = target_names[i]
                target_count = int(v)
                print(spacer + "return " + str(target_name) + \
                      " ( " + str(target_count) + " examples )")

    recurse(left, right, threshold, features, 0, 0)
    
    
def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")    




#####################################################




data=pd.read_csv('data.csv')
n = data.shape[0]
Y = data.shot_made_flag.values
#drop useless features
#data = data.drop('lat',1)
#data = data.drop('lon',1)

home = np.zeros(n)
away = np.ones(n)
for i in range(n):
    if('vs' in data['matchup'][i]):
        home[i] = 1
    else:
        home[i] = 0

data['Home'] = home


data['remaining_time'] = data['minutes_remaining'] * 60 + data['seconds_remaining']
data_rm = data.drop('team_id',1)	
data_rm = data_rm.drop('team_name',1)	
data_rm = data_rm.drop('game_event_id',1)
data_rm = data_rm.drop('game_id',1)
data_rm = data_rm.drop('minutes_remaining',1)
data_rm = data_rm.drop('seconds_remaining',1)
data_rm = data_rm.drop('shot_zone_area',1)
data_rm = data_rm.drop('shot_zone_range',1)
data_rm = data_rm.drop('shot_zone_basic',1)
#data_rm = data_rm.drop('shot_distance',1)
data_rm = data_rm.drop('game_date',1)
data_rm = data_rm.drop('matchup',1)
data_rm = data_rm.drop('shot_id',1)
data_rm = data_rm.drop('combined_shot_type',1)
data_rm = data_rm.drop('lat',1)
data_rm = data_rm.drop('loc_x',1)
data_rm = data_rm.drop('loc_y',1)
data_rm = data_rm.drop('period',1)
data_rm = data_rm.drop('playoffs',1)
data_rm = data_rm.drop('lon',1)
data_rm = data_rm.drop('shot_type',1)
data_rm = data_rm.drop('opponent',1)
data_rm = data_rm.drop('remaining_time',1)
data_rm = data_rm.drop('season',1)
#data_rm = data.drop('action_type',1)

#print("* data_rm.head()", data_rm.head(), sep="\n", end="\n\n")
#print("* data_rm.tail()", data_rm.tail(), sep="\n", end="\n\n")



(data_md, rep)= encode_target(data_rm, 'action_type')
(data_md, rep)= encode_target(data_md, 'shot_distance')
#(data_md, rep)= encode_target(data_md, 'home')

#(data_md, rep)= encode_target(data_md, 'shot_zone_area')
#(data_md, rep)= encode_target(data_md, 'shot_zone_range')
#(data_md, rep)= encode_target(data_md, 'season')
#(data_md, rep)= encode_target(data_md, 'remaining_time')
#(data_md, rep)= encode_target(data_md, 'game_date')
#(data_md, rep)= encode_target(data_md, 'opponent')
#(data_md, rep)= encode_target(data_md, 'playoffs')


#features = ['action_type','shot_zone_area','shot_zone_range','season','game_date']

#separate
index = np.zeros(5000)
j = 0
for i in range(n):
	if math.isnan(Y[i]):
		index[j] = i
		j = j + 1
  
D_test = data_md.ix[index,:]
D_train = data_md.drop(index, axis = 0)

X = D_train.drop('shot_made_flag',1)   
Y = D_train['shot_made_flag']                
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.25)

#clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=500, random_state=99)
clf = DecisionTreeClassifier(min_samples_split=400)
#graphic tree
clf.fit(X_train, Y_train) 
result_bit=clf.predict(X_test)
result_prop=clf.predict_proba(X_test)
'''
testing=result_bit
for i in range(0,6425):
    if result_prop[i,1]>0.4:
       testing[i]=1
    else:
       testing[i]=0
 
score=0
#for i in range(0,6424):
#    if testing[i,1]==pd.Series(Y_test)[i+1]:
#        score+=1
        
#score=score/6425
'''
score=clf.score(X_test, Y_test)

#with open("iris.dot", 'w') as f:
#    f = tree.export_graphviz(clf, out_file=f)
#dot_file = tree.export_graphviz(clf.tree_, out_file='tree_d1.dot', feature_names=X_train[features])  #export the tree to .dot file
#dot_file.close() #close that dot file.
 
#print("\n-- get_code:")
#get_code(dt, features, targets)
#visualize_tree(dt, features)
#visualize_tree(dt, features)
