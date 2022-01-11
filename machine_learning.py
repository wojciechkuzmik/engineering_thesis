import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, pair_confusion_matrix, rand_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, plot_importance

from prepare_data import prepare_data

pd.set_option('display.width', 600)
pd.set_option('max_columns', None)


def test_models(filename):
    X, y = prepare_data(filename)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)

    def get_confusion_matrix_values(y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        return cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    classifiers = {
        "DummyClassifier_stratified": DummyClassifier(strategy='stratified', random_state=0),
        "KNeighborsClassifier": KNeighborsClassifier(3),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "RandomForestClassifier": RandomForestClassifier(),
        "GradientBoostingClassifier": GradientBoostingClassifier(),
        "Perceptron": Perceptron(max_iter=40, eta0=0.1, random_state=1),
        "MLP": MLPClassifier(),
        "XGBClassifer tuned": XGBClassifier(colsample_bytree=0.8,
                                            gamma=0.9,
                                            max_depth=20,
                                            min_child_weight=1,
                                            scale_pos_weight=12,
                                            subsample=0.9,
                                            n_estimators=50,
                                            learning_rate=0.1,
                                            use_label_encoder=False),
        "LinearSVC": LinearSVC(max_iter=5000)
    }

    df_results = pd.DataFrame(columns=['model', 'accuracy', 'precision',
                                       'recall', 'f1', 'run_time', 'tp', 'fp',
                                       'tn', 'fn'])

    for key in classifiers:
        start_time = time.time()
        classifier = classifiers[key]
        model = classifier.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        run_time = format(round((time.time() - start_time) / 60, 2))
        tp, fp, fn, tn = get_confusion_matrix_values(y_test, y_pred)

        row = {'model': key,
               'accuracy': accuracy,
               'precision': precision,
               'recall': recall,
               'f1': f1,
               'run_time': run_time,
               'tp': tp,
               'fp': fp,
               'tn': tn,
               'fn': fn,
               }
        df_results = df_results.append(row, ignore_index=True)

    print(df_results.head(11))


def supervised_machine_learning(filename_learn, filename_test):
    X, y = prepare_data(filename_learn, print_corr=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)
    model = XGBClassifier(colsample_bytree=0.8,
                          gamma=0.9,
                          max_depth=20,
                          min_child_weight=1,
                          scale_pos_weight=12,
                          subsample=0.9,
                          n_estimators=50,
                          learning_rate=0.1,
                          use_label_encoder=False)
    model = model.fit(X_train, y_train)
    ax = plot_importance(model)
    fig = ax.figure
    fig.set_size_inches(10, 3)
    plt.show()
    y_pred = model.predict(X_test)
    print("\nroc_auc_score:")
    print(roc_auc_score(y_test, y_pred))
    print("\nClassification report for testing data:")
    print(classification_report(y_test, y_pred, labels=[1, 0], target_names=['match', 'not match']))

    print("\n\nweb scraping data results:")
    web_scraping_data, _ = prepare_data(filename_test)
    web_scraping_pred = model.predict(web_scraping_data)
    web_scraping_df = pd.read_csv(filename_test)
    web_scraping_df = web_scraping_df.drop(columns=["match"])
    web_scraping_df["pred"] = web_scraping_pred
    print(web_scraping_df.to_string(index=False))


def unsupervised_machine_learning(filename_learn, filename_test):
    X, y = prepare_data(filename_learn, print_corr=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)
    model = KMeans(n_clusters=2,
                   max_iter=100,
                   n_init=30)
    model.fit(X_train)
    y_pred = model.predict(X_test)
    print("\nPair confusion matrix for testing data:")
    print(pair_confusion_matrix(y_test, y_pred))
    print("\nRand index score for testing data:")
    print(rand_score(y_test, y_pred))

    print("\n\nweb scraping data results (unsupervised learning - 0, 1 are clusters): ")
    web_scraping_data, _ = prepare_data(filename_test)
    web_scraping_pred = model.predict(web_scraping_data)
    web_scraping_df = pd.read_csv(filename_test)
    web_scraping_df = web_scraping_df.drop(columns=["match"])
    web_scraping_df["pred"] = web_scraping_pred
    print(web_scraping_df.to_string(index=False))
