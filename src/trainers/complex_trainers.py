from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from src.features.customer_preprocessor import CustomPreprocessor
import joblib

class ModelSelector:
    def __init__(self):
        self.models = [
            ("Random Forest", RandomForestClassifier(random_state=42)),
            ("LightGBM", LGBMClassifier(random_state=42)),
            ("MLP", MLPClassifier(random_state=42, max_iter=2000, early_stopping=True))
        ]

        self.param_grids = [
            {
                'model__n_estimators': [100, 200, 500],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5, 10],
            },
            {
                'model__n_estimators': [100, 200, 500],
                'model__max_depth': [None, 10, 20],
                'model__learning_rate': [0.01,0.05, 0.1],
                'model__reg_alpha': [0, 0.1, 1],
                'model__reg_lambda': [0, 0.1, 1]
            },
            {
                'model__hidden_layer_sizes': [(100,), (100, 50), (50, 50)],
                'model__activation': ['relu', 'tanh'],
                'model__learning_rate_init': [0.001, 0.01, 0.05],
                'model__alpha': [0.0001, 0.001, 0.01],
                'model__solver': ['sgd', 'adam'],
            }
        ]

    def select_model(self, X_train, y_train):
        results = []
        
        for (name, model), param_grid in zip(self.models, self.param_grids):
            pipeline = Pipeline([
                ('preprocessor', CustomPreprocessor()),
                ('model', model)
            ])
    
            random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=10,
                                           scoring='roc_auc', cv=5, verbose=2, n_jobs=-1,random_state=42,
                                           error_score='raise')
            
            random_search.fit(X_train, y_train)
            
            result = {
                'name': name,
                'best_score': random_search.best_score_,
                'best_params': random_search.best_params_,
                'best_model': random_search.best_estimator_
            }
    
            # Save the best model to disk
            joblib.dump(result['best_model'], f'models/{name}.joblib')
    
            results.append(result)
            print(f"{name}: Best score = {result['best_score']}, Best Params: {result['best_params']}")
    
        sorted_results = sorted(results, key=lambda x: x['best_score'], reverse=True)
    
        return sorted_results

