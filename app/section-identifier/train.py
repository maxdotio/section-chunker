from sklearn.model_selection import GridSearchCV

def train_model(model, params, k, X, y):
    search_model = GridSearchCV(model, params, scoring="accuracy", cv=k, refit=True)
    best_model = search_model.fit(X,y).best_estimator_
    return best_model