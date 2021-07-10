import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


ord_enc = OrdinalEncoder()
df = pd.read_csv("data.csv")

headers = ["Award_Work","Care-taker_WhileWorking","SP_Supp","Using_DCC","ComfyLeave_ChildSick","MatPat_Leave","Flex_Whours","Eval_WorkProduced", "Prod"]
df["Award_Work"] = ord_enc.fit_transform(df[["Award_Work"]])
df["Care-taker_WhileWorking"] = ord_enc.fit_transform(df[["Care-taker_WhileWorking"]])
df["SP_Supp"] = ord_enc.fit_transform(df[["SP_Supp"]])
df["Using_DCC"] = ord_enc.fit_transform(df[["Using_DCC"]])
df["ComfyLeave_ChildSick"] = ord_enc.fit_transform(df[["ComfyLeave_ChildSick"]])
df["MatPat_Leave"] = ord_enc.fit_transform(df[["MatPat_Leave"]])
df["Flex_Whours"] = ord_enc.fit_transform(df[["Flex_Whours"]])
df["Eval_WorkProduced"] = ord_enc.fit_transform(df[["Eval_WorkProduced"]])
df["Prod"] = ord_enc.fit_transform(df[["Prod"]])
print(df.info())

features = ["Award_Work","Care-taker_WhileWorking","SP_Supp","Using_DCC","ComfyLeave_ChildSick","MatPat_Leave","Flex_Whours","Eval_WorkProduced"]
labels = ["Prod"]

X = df[features]
y = df[labels]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.30)

# function for fitting trees of various depths on the training data using cross-validation
def run_cross_validation_on_trees(X, y, tree_depths, cv=5, scoring='accuracy'):
    cv_scores_list = []
    cv_scores_std = []
    cv_scores_mean = []
    accuracy_scores = []
    
    for depth in tree_depths:
        tree_model = DecisionTreeClassifier(max_depth=depth)
        cv_scores = cross_val_score(tree_model, X, y, cv=cv, scoring=scoring)
        cv_scores_list.append(cv_scores)
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_std.append(cv_scores.std())
        accuracy_scores.append(tree_model.fit(X, y).score(X, y))
    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)
    accuracy_scores = np.array(accuracy_scores)
    print(accuracy_scores)
    print(cv_scores_mean)
    return cv_scores_mean, cv_scores_std, accuracy_scores
  
# function for plotting cross-validation results
def plot_cross_validation_on_trees(depths, cv_scores_mean, cv_scores_std, accuracy_scores, title):
    fig, ax = plt.subplots(1,1, figsize=(15,5))
    ax.plot(depths, cv_scores_mean, '-o', label='mean cross-validation accuracy', alpha=0.9)
    ax.fill_between(depths, cv_scores_mean-2*cv_scores_std, cv_scores_mean+2*cv_scores_std, alpha=0.2)
    ylim = plt.ylim()
    ax.plot(depths, accuracy_scores, '-*', label='train accuracy', alpha=0.9)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Tree depth', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(depths)
    ax.legend()

# fitting trees of depth 1 to 24
sm_tree_depths = range(1,6)
sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores = run_cross_validation_on_trees(X_train, y_train, sm_tree_depths)

# plotting accuracy
plot_cross_validation_on_trees(sm_tree_depths, sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores, 
                               'Accuracy per decision tree depth on training data')

