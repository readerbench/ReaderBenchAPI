from rb.similarity.vector_model import VectorModelType
from sklearn.model_selection import train_test_split
from sklearn import ensemble

import pandas as pd
import statistics
from sklearn.decomposition import PCA
import json
from os import path

from random import randrange

from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices
from rb.core.document import Document
from rb.core.lang import Lang
from rb.similarity.vector_model_factory import get_default_model, create_vector_model
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from scipy.stats import zscore

import numpy as np

import pickle


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
    # model = create_vector_model(Lang.RO, VectorModelType.WORD2VEC, 'readme', 300, False)
    doc = Document(Lang.RO, text)
    cna_graph = CnaGraph(docs=doc, models=[model])
    compute_indices(doc=doc, cna_graph=cna_graph)

    block = []

    for b in doc.get_blocks():
        sent = []
        for s in b.get_sentences():
            sent.append(s.text)
        block.append({'text': b.text, 'sentences': sent})

    feedback_text = {
        'doc': doc.text,
        'blocks': block
    }

    sentences = [sent.indices for sent in doc.get_sentences()]
    blocks = [block.indices for block in doc.get_blocks()]

    return {
            'text': feedback_text,
            'indices': {
                'document': doc.indices,
                'sentence': sentences,
                'block': blocks
            }
        }


def automatic_scoring(indices):
    url = 'rb_api/feedback/textual_indices.csv'
    X, y = prepare_dataset(url)
    # model = gradient_boosting_regression(X, y)
    with open("rb_api/feedback/gbr.pkl", "rb") as f:
        model = pickle.load(f)
    features = set(X.columns)
    used_features = {}
    indices = indices['indices']['document']
    for key, value in indices.items():
        if str(key) in features:
            used_features.update({str(key): [value]})
    used_feature_names = used_features.keys()
    for key in features:
        if key not in used_feature_names:
            used_features.update({str(key): [0]})
    new_doc_df = pd.DataFrame.from_dict(used_features)
    prediction = model.predict(new_doc_df)
    return round(prediction[0], 2)


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
    features_to_be_eliminated = []

    textual_indices_df = dataset.drop(columns=['Title', 'Subject', 'Grade'])
    # normalize data
    textual_indices_df = textual_indices_df.apply(zscore)
    skew = textual_indices_df.skew()
    kurtosis = textual_indices_df.kurtosis()

    for index, value in enumerate(skew):
        if skew[index] > 4 or kurtosis[index] > 11:
            features_to_be_eliminated.append(textual_indices_df.columns[index + 3])

    return dataset.drop(columns=features_to_be_eliminated)


def remove_corelated_indices(dataset):
    textual_indices_df = dataset.drop(columns=['Title', 'Subject', 'Grade'])

    # normalize data
    textual_indices_df = textual_indices_df.apply(zscore)

    pearson_corelation = textual_indices_df.corr(method='pearson')
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


def filter_dataset_PCA():
    if path.exists('rb_api/feedback/pca-dataset.pkl'):
        print('pca-dataset exists')
        with open('rb_api/feedback/pca-dataset.pkl', 'rb') as f:
            dataset = pickle.load(f)
    else:
        print('creating pca-dataset.pkl')
        url = 'rb_api/feedback/textual_indices.csv'
        dataset = pd.read_csv(url)
        # dataset = dataset.drop(columns=['Unnamed: 0'])
        dataset = filter_rare(dataset)
        dataset = remove_outliers(dataset)
        dataset = prune_Skewness_Kurtosis(dataset)
        dataset = remove_corelated_indices(dataset)

        pickle.dump(dataset, open('rb_api/feedback/pca-dataset.pkl', 'wb'))

    return dataset


def create_PCA(dataset):
    if path.exists('rb_api/feedback/pca.pkl') and path.exists('rb_api/feedback/pca-components.pkl'):
        print('pca & pca-components exist')
        with open('rb_api/feedback/pca.pkl', 'rb') as f:
            pca = pickle.load(f)
        with open('rb_api/feedback/pca-components.pkl', 'rb') as f:
            principalComponents = pickle.load(f)

    else:
        textual_indices_df = dataset.drop(columns=['Title', 'Subject', 'Grade'])
        pca = PCA(n_components=4)
        print('creating pca.pkl')
        pickle.dump(pca, open('rb_api/feedback/pca.pkl', 'wb'))
        principalComponents = pca.fit_transform(textual_indices_df)
        print('creating pca-components.pkl')
        pickle.dump(principalComponents, open('rb_api/feedback/pca-components.pkl', 'wb'))

    principal_data_frame = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    principal_data_frame.insert(0, 'Title', dataset['Title'], True)
    principal_data_frame.insert(1, 'Subject', dataset['Subject'], True)
    principal_data_frame.insert(2, 'Grade', dataset['Grade'], True)

    return pca, principal_data_frame


def compute_PCA_on_new_text(doc_indices):
    dataset = filter_dataset_PCA()
    pca, pca_dataset = create_PCA(dataset)

    features = set(dataset.columns)
    indices = {}
    for key, value in doc_indices.items():
        if str(key) in features:
            indices.update({str(key): [value]})
    dataFrame = pd.DataFrame.from_dict(indices)
    dataset = dataset.drop(columns=['Title', 'Subject', 'Grade'])
    dataset = dataset.append(dataFrame)
    dataset.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    principalComponents = pca.fit_transform(dataset)
    principal_data_frame = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    pca_df = principal_data_frame.tail(1).values.tolist()[0]
    pca_result = {'PC1': pca_df[0], 'PC2': pca_df[1], 'PC3': pca_df[2], 'PC4': pca_df[3]}
    return pca_result


def get_feedback_metrics(url):
    with open(url, encoding='UTF-8') as f:
        feedback_metrics = json.load(f)
    return feedback_metrics


def automatic_feedback_granularity(doc_indices, granularity, feedback_metrics):
    indices = {}
    feedback = []
    for key, value in doc_indices.items():
        indices.update({str(key): value})

    for metric in feedback_metrics[granularity]:
        if indices[metric['id']] <= metric['min']:
            feedback.append({
                'name': metric['name'],
                'description': metric['feedbackMessagesLow'][randrange(len(metric['feedbackMessagesLow']))]
            })
        if indices[metric['id']] >= metric['max']:
            feedback.append({
                'name': metric['name'],
                'description': metric['feedbackMessagesHigh'][randrange(len(metric['feedbackMessagesHigh']))]
            })
    return feedback


def automatic_feedback_pca(doc_indices, feedback_metrics):
    feedback = []
    pca_indices = compute_PCA_on_new_text(doc_indices)
    for metric in feedback_metrics['pca']:
        if pca_indices[metric['id']] <= metric['min']:
            feedback.append({
                'name': metric['name'],
                'description': metric['feedbackMessagesLow'][randrange(len(metric['feedbackMessagesLow']))]
            })
        if pca_indices[metric['id']] >= metric['max']:
            feedback.append({
                'name': metric['name'],
                'description': metric['feedbackMessagesHigh'][randrange(len(metric['feedbackMessagesHigh']))]
            })
    return feedback

def automatic_feedback(doc_indices):
    url = 'rb_api/feedback/feedback_rules.json'
    feedback_metrics = get_feedback_metrics(url)

    return {
        'text': doc_indices['text'],
        'document': automatic_feedback_granularity(doc_indices['indices']['document'], 'document', feedback_metrics),
        'sentence': [automatic_feedback_granularity(ind, 'sentence', feedback_metrics) for ind in doc_indices['indices']['sentence']],
        'block': [automatic_feedback_granularity(ind, 'block', feedback_metrics) for ind in doc_indices['indices']['block']],
        'pca': automatic_feedback_pca(doc_indices['indices']['document'], feedback_metrics)
    }


if __name__ == '__main__':
    text = u'Povestea lui Harap-Alb este un basm cult de Ion Creangă, care reprezintă o sinteză de motive epice cu o circulaţie foarte largă. Respectând tiparul basmului, textul începe cu o formulă introductivă: Amu cică era odată, care avertizează cititorul asupra intrării într-o lume a poveştii. Spre deosebire de basmele populare, unde formula introductivă este compusă din trei termeni, unul care atestă o existenţă, (a fost odată), altul care o neagă (ca niciodată) şi, cel din urmă, format dintr-o serie de complemente circumstanţiale de timp care induc fantasticul, aici intrarea ex abrupto în text: era odată un craiu, care avea trei feciori… situează deocamdată textul la intersecţia dintre povestire şi basm. Structura textului corespunde basmului. Lipsa este marcată de scrisoarea lui Verde- împărat şi se concretizează în absenţa bărbatului, de aceea el îl roagă pe fratele lui să i-1 trimită pe cel mai bun dintre băieţi ca să rămână urmaş la tron. Următoarea etapă este căutarea eroului. în basm, ea se concretizează prin încercarea la care îşi supune craiul băieţii: se îmbracă în piele de urs şi iese în faţa lor de sub un pod. Conform structurii formale a basmului cel care reuşeşte să treacă proba este fiul cel mic; el trece proba din două motive: primul, se înscrie în etapele iniţierii cu ajutorul dat de Sfânta Duminecă, care îi spune să ia armele tatălui şi calul care va veni la tava cu jăratic; al doilea este de natură personală. El devine protagonistul acţiunii. Fiul cel mic este curajos. Podul reprezintă, în plan simbolic, limita lumii cunoscute – lumea împărăţiei craiului unde codul comportamental este bine cunoscut de fiul cel mic – şi punctul iniţial al unui spaţiu necunoscut. De aceea tatăl îi dă în acest loc primele indicaţii despre noua lume: să se ferească de omul spân şi de împăratul Roş şi îi dă piele de urs. Din acest moment debutează a doua etapă a basmului: înşelătoria. Pe drum, fiul cel mic al craiului se întâlneşte cu un om spân care îi cere să-1 ia în slujba lui. Băiatul refuză de două ori, dar a treia oară spânul reuşeşte să-1 înşele: ajunşi la o fântână, spânul intră şi se răcoreşte, apoi îl sfătuieşte pe băiat să facă acelaşi lucru. Fiul craiului, boboc tn felul său la trebi de aieste, se potriveşte Spânului, şi se bagă în fântână, fără să-l trăsnească prin minte ce i se poate întâmpla. Momentul este important pentru imaginea fiului de crai dinaintea încercărilor. Trăsătura vizată este naivitatea, trăsătură marcată direct de autor – fiind boboc la trebi din aiestea, Harap-Alb nu intuieşte că Spânul, antagonistul său, are intenţii ascunse. Naivitatea eroului e foarte importantă în evoluţia conflictului, întrucât textul urmăreşte tocmai maturizarea lui Harap-Alb. Naivitatea se înscrie în codul ritual al iniţierii prin care trece fiul craiului. Atitudinea empatică a naratorului este menită să sporească tensiunea dramatică şi să inducă un principiu etic.'

    indices = compute_textual_indices(text)
    print(automatic_scoring(indices))