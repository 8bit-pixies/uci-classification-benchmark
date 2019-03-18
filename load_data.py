import pandas as pd
from sklearn.linear_model import SGDClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score

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
    name = 'adult'
    name = 'dna'
    X, y = load_data(name)
    X_test, y_test = load_data(name, 'test')
    model = SGDClassifier(tol=1e-3, max_iter=1000)
    model.fit(X, y)
    print(model.score(X, y))
    print(model.score(X_test, y_test))

    print("\nTraining LightGBM")
    meta = [x for x in meta_data if x['dataset'] == name][0]
    objective = "binary" if meta['nclass'] == 2 else 'multiclass'
    gbm_config = dict(
        boosting_type = 'gbdt',
        objective="binary",
        num_leaves=256,
        n_estimators=10
    )

    gbm = lgb.LGBMClassifier(**gbm_config)
    gbm.fit(X, y)
    print(accuracy_score(y, model.predict(X)))
    print(accuracy_score(y_test, model.predict(X_test)))


