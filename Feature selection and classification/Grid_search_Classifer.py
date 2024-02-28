from sklearn.svm import SVC					       
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# KNN
param_grid_knn = {'n_neighbors': [5, 10, 15, 20]}

def train_knn_with_grid_search(x_train, y_train, param_grid,cv=5):
    
    grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv)
    grid_search_knn.fit(x_train, y_train)
    
    best_params = grid_search_knn.best_params_
    
    model = KNeighborsClassifier(**best_params)
    model.fit(x_train, y_train)

    return model, best_params

# SVM
param_grid_svm = {'C': [0.1, 1, 10],
                  'kernel': ['linear', 'rbf', 'poly'],
                  'gamma': ['scale', 'auto', 0.1, 0.5, 1.0]}

def train_svm_with_grid_search(x_train, y_train, param_grid, cv=5):

    grid_search_svm = GridSearchCV(SVC(), param_grid, cv=cv)
    grid_search_svm.fit(x_train, y_train)

    best_params = grid_search_svm.best_params_

    model = SVC(**best_params)
    model.fit(x_train, y_train)

    return model, best_params

# LR
param_grid_lr = {'C': [0.1, 1, 10]}

def train_lr_with_grid_search(x_train, y_train, param_grid, cv=5):

    grid_search_lr = GridSearchCV(LogisticRegression(), param_grid, cv=cv)
    grid_search_lr.fit(x_train, y_train)

    best_params = grid_search_lr.best_params_

    model = LogisticRegression(**best_params)
    model.fit(x_train, y_train)

    return model, best_params

# RF
param_grid_rf = {'n_estimators': [50, 100, 150],
                 'max_depth': [None, 10, 20],
                 'min_samples_split': [2, 5, 10],
                 'min_samples_leaf': [1, 2, 4]}

def train_rf_with_grid_search(x_train, y_train, param_grid, cv=5):
   
    grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid, cv=cv)
    grid_search_rf.fit(x_train, y_train)

    best_params = grid_search_rf.best_params_

    model = RandomForestClassifier(**best_params)
    model.fit(x_train, y_train)

    return model, best_params

# GB
param_grid_gb = {'n_estimators': [50,75],
                 'max_depth': [5],
                 'learning_rate': [0.1, 0.5]}

def train_gb_with_grid_search(x_train, y_train, param_grid, cv=5):

    grid_search_gb = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=cv)
    grid_search_gb.fit(x_train, y_train)

    best_params = grid_search_gb.best_params_

    model = GradientBoostingClassifier(**best_params)
    model.fit(x_train, y_train)

    return model, best_params

# ERT
param_grid_ert = {'n_estimators': [50, 75],
                  'max_depth': [10],
                  'min_samples_split': [5, 10],
                  'min_samples_leaf': [2, 4]}

def train_ert_with_grid_search(x_train, y_train, param_grid, cv=5):

    grid_search_ert = GridSearchCV(ExtraTreesClassifier(), param_grid, cv=cv)
    grid_search_ert.fit(x_train, y_train)

    best_params = grid_search_ert.best_params_

    model = ExtraTreesClassifier(**best_params)
    model.fit(x_train, y_train)

    return model, best_params

# XGB
param_grid_xgb = {'n_estimators': [30, 50,70,90,120],
                   'max_depth': [4, 6, 8, 11, 13],
                   'learning_rate': [0.01, 0.03, 0.06, 0.08, 0.1],
                   'reg_alpha':[None, 1, 2],
                   'reg_lambda':[None, 2, 4]}

def train_xgb_with_grid_search(x_train, y_train, param_grid, cv=5):

    grid_search_xgb = GridSearchCV(XGBClassifier(), param_grid, cv=cv)
    grid_search_xgb.fit(x_train, y_train)

    best_params = grid_search_xgb.best_params_

    model = XGBClassifier(**best_params)

    return model, best_params

#AB
param_grid_ab = {'n_estimators': [50, 75, 100],
                 'learning_rate': [0.01, 0.1, 0.3, 0.5]}

def train_ab_with_grid_search(x_train, y_train, param_grid, cv=5):

    base_classifier = DecisionTreeClassifier(max_depth=3)  # 基础分类器，可以根据需要更改
    grid_search_ab = GridSearchCV(AdaBoostClassifier(base_classifier), param_grid, cv=cv)
    grid_search_ab.fit(x_train, y_train)

    best_params = grid_search_ab.best_params_

    model = AdaBoostClassifier(base_classifier, **best_params)
    model.fit(x_train, y_train)

    return model, best_params