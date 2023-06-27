from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from src.features.customer_preprocessor import CustomPreprocessor
import joblib
import os

class Trainer:
    def __init__(self):
        self.pipeline = Pipeline([
            ('preprocessor', CustomPreprocessor()),
            ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=2000))
        ])
        
        self.gs_param_grid = {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear', 'saga']  # solvers that support both 'l1' and 'l2' penalties
        }

    def train(self, X_train, y_train, model_filename='model.joblib'):
        grid_search = GridSearchCV(self.pipeline, self.gs_param_grid, scoring='roc_auc', cv=5, verbose=2, n_jobs=-1, error_score='raise')
        grid_search.fit(X_train, y_train)
        model_filepath = os.path.join("models", model_filename)
        joblib.dump(grid_search, model_filepath)
        return grid_search


