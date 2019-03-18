import pandas as pd
from sklearn.linear_model import SGDClassifier

meta_data = [{'dataset': 'adult', 'nclass': 2, 'label': 'income'},
 {'dataset': 'covtype', 'nclass': 7, 'label': 'X55'},
 {'dataset': 'dna', 'nclass': 3, 'label': 'Class'},
 {'dataset': 'glass', 'nclass': 6, 'label': 'Class'},
 {'dataset': 'letter', 'nclass': 26, 'label': 'lettr'},
 {'dataset': 'sat', 'nclass': 8, 'label': 'X37'},
 {'dataset': 'shuttle', 'nclass': 7, 'label': 'Class'},
 {'dataset': 'simplemandelon', 'nclass': 2, 'label': 'y'},
 {'dataset': 'soybean', 'nclass': 19, 'label': 'Class'},
 {'dataset': 'yeast', 'nclass': 10, 'label': 'yeast'}
]

def split_pred(df, label):
    return df[[x for x in df.columns if x != label]], df[label]

def load_data(name, subset='train'):
    dataset = pd.read_csv(f"clean_data/{name}_{subset}_scale.csv")
    meta = [x for x in meta_data if x['dataset'] == name][0]
    X, y = split_pred(dataset, meta['label'])
    return X, y

if __name__ == '__main__':
    X, y = load_data('adult')
    X_test, y_test = load_data('adult', 'test')
    model = SGDClassifier(tol=1e-3, max_iter=1000)
    model.fit(X, y)
    model.score(X, y)
    model.score(X_test, y_test)

