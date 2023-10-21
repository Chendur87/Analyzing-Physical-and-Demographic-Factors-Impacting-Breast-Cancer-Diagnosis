from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np


class ResearchQuestionThree:
    '''
    Runs the functions pertaining to my third research question:

    How well can Machine Learning determine if a patient has breast cancer
    (malignant vs. benign tumors) based on the features of the tumor? Can we
    identify optimal hyperparameters to create an optimal model and determine
    the top features that contribute to the decisions of the model?
    '''

    def __init__(self):
        # dataset from ResearchQuestionOne
        self.df1 = pd.read_csv('datasets/data.csv')
        self.df1 = self.df1.loc[:, 'diagnosis':'symmetry_mean']
        # third dataset reserved for ML
        self.df3 = pd.read_csv('datasets/breast_cancer_bd.csv')
        self.df3 = self.df3.rename(columns={'Class': 'diagnosis'})
        # making 2 → B and 4 → M to not confuse the model with
        # quantitative diagnosis (which doesn't make sense)
        self.df3.loc[self.df3['diagnosis'] == 2, 'diagnosis'] = 'B'
        self.df3.loc[self.df3['diagnosis'] == 4, 'diagnosis'] = 'M'
        self.df3 = self.df3.replace('?', np.nan)
        self.df3 = self.df3.dropna()

    def run(self):
        """
        Runs the method for maximizing the test_accuracy for each df
        """
        for index, df in enumerate([self.df1, self.df3]):
            self.maximizing_test_accuracy(df, 2 * index + 1)

    def maximizing_test_accuracy(self, df, dataset_num):
        """
        Given the paramaters to tune the model, this function will find 
        the most optimal parameters to achieve the highest test accuracy
        """
        # Separate the df into features and labels
        # features: tumor specifics
        # label: whether the patient has breast cancer or not
        features = df.loc[:, df.columns != 'diagnosis']
        labels = df['diagnosis']

        rfc = RandomForestClassifier()
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'max_features': ['sqrt', 'log2', None]
        }

        # train RandomForestClassifier with all the parameter combinations
        # (3*3*3 = 27) to find the most optimal model from those parameters
        # utilize GridSearchCV to create 5 folds in the data, where 4 folds
        # will be the data that the RandomForestClassifier would be trained
        # on, and the last fold will act as the testing data
        # each of the folds will "take turns" in becoming the test data for
        # that cross validation aspect, creating the most optimal model
        # based on the folds and the parameters the model can utilize
        grid_search = GridSearchCV(rfc, param_grid, cv=5)
        grid_search.fit(features, labels)

        print(f'\nFor Dataset #{dataset_num}')
        print(f'Best Parameters Found {grid_search.best_params_}')
        print(f'Best Accuracy Found {grid_search.best_score_}')

        # utilize the best parameters in the model to determine what features
        # are the top contributors
        self.determining_top_features(df, grid_search.best_params_)

    def determining_top_features(self, df, best_params):
        """
        Figures out the top features that contribute to the model given the
        optimal parameters from maximizing_test_accuracy
        """
        # split the features and labels into training and test data
        features_train, features_test, labels_train, labels_test = train_test_split(
            df.loc[:, df.columns != 'diagnosis'],
            df['diagnosis'],
            test_size=0.2)

        # Determine the top 3 features with the highest ANOVA F-values
        # (another statistical test) using the f_classif score function
        # fit the SelectKBest for those features
        feature_selector = SelectKBest(score_func=f_classif, k=3)
        feature_selector.fit(features_train, labels_train)

        # After fitting the SelectKBest, transform the datasets to only include
        # the top 3 features
        features_train_selected = feature_selector.transform(features_train)
        features_test_selected = feature_selector.transform(features_test)

        # create a model with the optimal parameters given from the previous
        # method
        rf = RandomForestClassifier(**best_params)

        # train the model with those selected features
        rf.fit(features_train_selected, labels_train)

        accuracy = rf.score(features_test_selected, labels_test)
        print(f'Accuracy when top features are selected:{accuracy}')
        # .get_support() gets the top 3 features that contributed to the model
        print(df.loc[:, df.columns != 'diagnosis'].columns[
            feature_selector.get_support()])