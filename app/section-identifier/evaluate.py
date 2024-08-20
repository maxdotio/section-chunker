from sklearn.metrics import classification_report

def make_eval_report(model,X_test,y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test,y_pred)
    print(report)