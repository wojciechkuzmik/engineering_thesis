import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, pair_confusion_matrix, rand_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from prepare_data import prepare_data

pd.set_option('display.width', 600)
pd.set_option('max_columns', None)


def test_models(filename):
    X, y = prepare_data(filename)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)

    def get_confusion_matrix_values(y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        return cm[1][1], cm[0][1], cm[1][0], cm[0][0]

    classifiers = {
        "DummyClassifier_stratified": DummyClassifier(strategy='stratified'),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "RandomForestClassifier": RandomForestClassifier(),
        "GradientBoostingClassifier": GradientBoostingClassifier(),
        "Perceptron": Perceptron(),
        "MLP": MLPClassifier()
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


def tune_gbclassifier(filename):
    X, y = prepare_data(filename)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)
    print("""Choose the parameter to tune:
    1. learning rate
    2. number of estimators
    3. max depth
    4. min samples split
    5. min samples leaf
    6. max features
    """)
    option = input("Enter an option (only number): ")
    if option == "1":
        learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
        train_results = []
        test_results = []
        for eta in learning_rates:
            model = GradientBoostingClassifier(learning_rate=eta)
            model.fit(x_train, y_train)
            train_pred = model.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            y_pred = model.predict(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)

        line1, = plt.plot(learning_rates, train_results, 'b', label ="Train AUC")
        line2, = plt.plot(learning_rates, test_results, 'r', label ="Test AUC")
        plt.ylabel('AUC score')
        plt.xlabel('learning rate')
        plt.show()
    elif option == "2":
        n_estimators = [1, 2, 16, 32, 64, 200, 500, 1000]
        train_results = []
        test_results = []
        for estimator in n_estimators:
            model = GradientBoostingClassifier(n_estimators=estimator)
            model.fit(x_train, y_train)
            train_pred = model.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            y_pred = model.predict(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
        line1, = plt.plot(n_estimators, train_results, 'b', label ='Train AUC')
        line2, = plt.plot(n_estimators, test_results, 'r', label ='Test AUC')
        plt.ylabel('AUC score')
        plt.xlabel('n_estimators')
        plt.show()
    elif option == "3":
        max_depths = np.linspace(1, 32, 32, endpoint=True)
        train_results = []
        test_results = []
        for max_depth in max_depths:
            model = GradientBoostingClassifier(max_depth=max_depth)
            model.fit(x_train, y_train)
            train_pred = model.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            y_pred = model.predict(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
        line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
        line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')
        plt.ylabel('AUC score')
        plt.xlabel('Tree depth')
        plt.show()
    elif option == "4":
        min_samples_splits = np.linspace(0.01, 1.0, 10, endpoint=True)
        train_results = []
        test_results = []
        for min_samples_split in min_samples_splits:
            model = GradientBoostingClassifier(min_samples_split=min_samples_split)
            model.fit(x_train, y_train)
            train_pred = model.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            y_pred = model.predict(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
        line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')
        line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')
        plt.ylabel('AUC score')
        plt.xlabel('min samples split')
        plt.show()
    elif option == "5":
        min_samples_leafs = np.linspace(0.01, 0.5, 5, endpoint=True)
        train_results = []
        test_results = []
        for min_samples_leaf in min_samples_leafs:
            model = GradientBoostingClassifier(min_samples_leaf=min_samples_leaf)
            model.fit(x_train, y_train)
            train_pred = model.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            y_pred = model.predict(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
        line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train AUC')
        line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test AUC')
        plt.ylabel('AUC score')
        plt.xlabel('min samples leaf')
        plt.show()
    elif option == "6":
        max_features = list(range(1, x_train.shape[1]))
        train_results = []
        test_results = []
        for max_feature in max_features:
            model = GradientBoostingClassifier(max_features=max_feature)
            model.fit(x_train, y_train)
            train_pred = model.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            y_pred = model.predict(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
        line1, = plt.plot(max_features, train_results, 'b', label='Train AUC')
        line2, = plt.plot(max_features, test_results, 'r', label='Test AUC')
        plt.ylabel('AUC score')
        plt.xlabel('max features')
        plt.show()
    else:
        print("wrong option selcted")
        pass


def supervised_machine_learning(filename_learn, filename_test):
    X, y = prepare_data(filename_learn, print_corr=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)
    model = GradientBoostingClassifier(learning_rate=0.5, n_estimators=300, max_depth=5, min_samples_split=0.01,
                                       min_samples_leaf=0.01, max_features=4)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nroc_auc_score:")
    print(roc_auc_score(y_test, y_pred))
    print("\nClassification report for testing data:")
    print(classification_report(y_test, y_pred, labels=[1, 0], target_names=['match', 'not match']))

    print("\n\nweb scraping data results:")
    web_scraping_data, _ = prepare_data(filename_test)
    web_scraping_pred = model.predict_proba(web_scraping_data)[:, 1]
    web_scraping_df = pd.read_csv(filename_test)
    web_scraping_df = web_scraping_df.drop(columns=["label"])
    web_scraping_df["pred"] = web_scraping_pred
    print(web_scraping_df.to_string(index=False))


def unsupervised_machine_learning(filename_learn, filename_test):
    X, y = prepare_data(filename_learn, print_corr=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)
    model = KMeans(n_clusters=2,
                   max_iter=100,
                   n_init=10)
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
    web_scraping_df = web_scraping_df.drop(columns=["label"])
    web_scraping_df["pred"] = web_scraping_pred
    print(web_scraping_df.to_string(index=False))
