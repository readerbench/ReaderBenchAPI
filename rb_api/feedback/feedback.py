import json
import pickle
import statistics

import numpy as np
import pandas as pd
from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices
from rb.core.document import Document
from rb.core.lang import Lang
from rb.similarity.vector_model_factory import get_default_model
from sklearn import ensemble
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.model_selection import train_test_split


def prepare_dataset(filename):
    dataset = pd.read_csv(filename)
    dataset = dataset.sample(frac=1).reset_index(drop=True)  # shuffle entries
    y = dataset['Grade']
    y = y.map(lambda elem: int(round(elem)))
    X = dataset.drop(['Title', 'Subject', 'Grade'], axis=1)
    return X, y


def gradient_boosting_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    gbr = ensemble.GradientBoostingRegressor(learning_rate=0.1, n_estimators=100)
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)

    MeanSquaredError = mean_squared_error(y_test, y_pred)
    VarinaceScore = gbr.score(X_test, y_test)
    CohenKappaScore = cohen_kappa_score(np.rint(y_pred), np.rint(y_test))
    return gbr, MeanSquaredError, VarinaceScore, CohenKappaScore


def compute_textual_indices(text):
    model = get_default_model(Lang.RO)
    doc = Document(Lang.RO, text)
    cna_graph = CnaGraph(docs=doc, models=[model])
    compute_indices(doc=doc, cna_graph=cna_graph)
    return doc.indices


def automatic_scoring(doc_indices):
    url = 'rb_api/feedback/textual_indices.csv'
    X, y = prepare_dataset(url)
    with open("rb_api/feedback/gbr.pkl", "rb") as f:
        model = pickle.load(f)
    # model, _, _, _ = gradient_boosting_regression(X, y)
    features = set(X.columns)
    used_features = {}
    for key, value in doc_indices.items():
        if str(key) in features:
            used_features.update({str(key): [value]})
    new_doc_df = pd.DataFrame.from_dict(used_features)
    prediction = model.predict(new_doc_df)
    return int(round(prediction[0]))


# remove features that have more than 20% of values equal to 0 or -1
def filter_rare(dataset):
    # first 3 features are 'Title', 'Subject' and 'Grade'; the rest are textual indices
    features = dataset.columns[3:]
    features_to_be_eliminated = []
    for feature in features:
        textual_indices = dataset[feature].values.tolist()
        zeros = textual_indices.count(0)
        negones = textual_indices.count(-1)
        if zeros >= 0.2 * len(textual_indices) or negones >= 0.2 * len(textual_indices):
            features_to_be_eliminated.append(feature)
    return dataset.drop(columns=features_to_be_eliminated)


# remove outlier documets
def remove_outliers(dataset):
    features = dataset.columns[3:]
    docs_to_be_eliminated = []
    metrics = {}
    for feature in features:
        textual_indices = dataset[feature].values.tolist()
        mean = statistics.mean(textual_indices)
        std = statistics.stdev(textual_indices)
        metrics[feature] = {'mean': mean, 'std': std}
    for row_number, row in dataset.iterrows():
        number_of_outlier_features = 0
        for elem in zip(features, row[3:]):
            (feature, value) = elem
            if value < metrics[feature]['mean'] - 2 * metrics[feature]['std']:
                number_of_outlier_features += 1
        if number_of_outlier_features >= 0.1 * len(features):
            docs_to_be_eliminated.append(row_number)
    return dataset.drop(dataset.index[docs_to_be_eliminated])


def prune_Skewness_Kurtosis(dataset):
    features = dataset.columns[3:]
    features_to_be_eliminated = []
    for feature in features:
        textual_indices = dataset[feature]
        skewness_score = textual_indices.skew()
        kurtosis_score = textual_indices.kurtosis()
        if skewness_score > 2 or kurtosis_score > 3.5:
            features_to_be_eliminated.append(feature)
    return dataset.drop(columns=features_to_be_eliminated)


def remove_corelated_indices(dataset):
    textual_indidces_df = dataset.drop(columns=['Title', 'Subject', 'Grade'])
    pearson_corelation = textual_indidces_df.corr(method='pearson')
    features = pearson_corelation.columns
    corelations = {}
    features_to_be_eliminated = []
    for column in features:
        for row in features:
            if column == row:
                continue
            if pearson_corelation[column][row] > 0.9:
                if row not in corelations.keys():
                    corelations[row] = [column]
                else:
                    if column not in corelations[row]:
                        corelations[row].append(column)
    corelations = dict(sorted(corelations.items(), key=lambda item: len(item[1]), reverse=True))
    for key in corelations:
        if len(corelations[key]) == 0:
            continue
        can_remove = True
        for feature in corelations[key]:
            if feature in features_to_be_eliminated and len(corelations[feature]) == 1 and key in corelations[feature]:
                can_remove = False
                break
        if can_remove:
            features_to_be_eliminated.append(key)
            for feature in corelations[key]:
                corelations[feature] = list(filter(lambda x: x != key, corelations[feature]))
    return dataset.drop(columns=features_to_be_eliminated)


def create_PCA(dataset):
    textual_indidces_df = dataset.drop(columns=['Title', 'Subject', 'Grade'])
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(textual_indidces_df)
    principal_data_frame = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3'])
    principal_data_frame.insert(0, 'Title', dataset['Title'], True)
    principal_data_frame.insert(1, 'Subject', dataset['Subject'], True)
    principal_data_frame.insert(2, 'Grade', dataset['Grade'], True)
    return pca, principal_data_frame


def compute_PCA_on_new_text(doc_indices):
    url = 'rb_api/feedback/pca_textual_indices.csv'
    dataset = pd.read_csv(url)
    dataset = dataset.drop(columns=['Unnamed: 0'])
    # dataset = filter_rare(dataset)
    # dataset = remove_outliers(dataset)
    # dataset = prune_Skewness_Kurtosis(dataset)
    # dataset = remove_corelated_indices(dataset)
    # dataset.to_csv('rb_api/feedback/pca_textual_indices.csv')
    # pca, pca_dataset = create_PCA(dataset)
    with open('rb_api/feedback/pca.pkl', "rb") as f:
        pca = pickle.load(f)
    features = set(dataset.columns)
    indices = {}
    for key, value in doc_indices.items():
        if str(key) in features:
            indices.update({str(key): [value]})
    dataFrame = pd.DataFrame.from_dict(indices)
    dataset = dataset.drop(columns=['Title', 'Subject', 'Grade'])
    dataset = dataset.append(dataFrame)
    principalComponents = pca.fit_transform(dataset)
    principal_data_frame = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3'])
    pca_df = principal_data_frame.tail(1).values.tolist()[0]
    pca_result = {'PC1': pca_df[0], 'PC2': pca_df[1], 'PC3': pca_df[2]}
    return pca_result


def get_feedback_metrics(url):
    with open(url, encoding="UTF-8") as f:
        feedback_metrics = json.load(f)
    return feedback_metrics


def automatic_feedback(doc_indices):
    url = 'rb_api/feedback/feedback_rules.json'
    feedback_metrics = get_feedback_metrics(url)
    indices = {}
    for key, value in doc_indices.items():
        indices.update({str(key): value})
    feedback = []
    for metric in feedback_metrics['document']:
        if indices[metric['id']] <= metric['min']:
            feedback.append({
                'name': metric['name'],
                'description': metric['feedbackMessagesLow']
            })
        if indices[metric['id']] >= metric['max']:
            feedback.append({
                'name': metric['name'],
                'description': metric['feedbackMessagesHigh']
            })

    pca_indices = compute_PCA_on_new_text(doc_indices)
    for metric in feedback_metrics['pca']:
        if pca_indices[metric['id']] <= metric['min']:
            feedback.append({
                'name': metric['name'],
                'description': metric['feedbackMessagesLow']
            })
        if pca_indices[metric['id']] >= metric['max']:
            feedback.append({
                'name': metric['name'],
                'description': metric['feedbackMessagesHigh']
            })
    return feedback
