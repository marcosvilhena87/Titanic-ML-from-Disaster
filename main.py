# Simple baseline model for Kaggle Titanic competition
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, GridSearchCV
from joblib import dump


def add_title_column(df):
    """Extract honorific titles from the Name field."""
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
    return df


def load_data():
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def preprocess_data(train_df, test_df, include_title=False, model_type="logistic"):
    y = train_df['Survived']

    for df in (train_df, test_df):
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        if include_title:
            add_title_column(df)

    X = train_df.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
    X_test = test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    numeric_features = [
        'Age',
        'SibSp',
        'Parch',
        'Fare',
        'FamilySize',
        'IsAlone',
    ]
    categorical_features = ['Pclass', 'Sex', 'Embarked']
    if include_title:
        categorical_features.append('Title')

    numeric_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='median'))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    if model_type == "logistic":
        classifier = LogisticRegression(max_iter=1000, solver='liblinear')
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    elif model_type == "gradient_boosting":
        classifier = GradientBoostingClassifier(random_state=0)
    else:
        raise ValueError(f"Modelo desconhecido: {model_type}")

    clf = Pipeline(
        steps=[('preprocessor', preprocessor),
               ('classifier', classifier)]
    )

    return X, y, X_test, clf


def train_and_predict(X, y, X_test, clf):
    clf.fit(X, y)
    predictions = clf.predict(X_test)
    probabilities = clf.predict_proba(X_test)[:, 1]
    return predictions, probabilities


def load_previous_submissions():
    """Load historic submissions and their scores."""
    scores_path = os.path.join('submissions', 'scores.csv')
    if not os.path.exists(scores_path):
        return []
    scores_df = pd.read_csv(scores_path, sep=';')
    submissions = []
    for _, row in scores_df.iterrows():
        sub_path = os.path.join('submissions', f"{row['Submission']}.csv")
        if os.path.exists(sub_path):
            df = pd.read_csv(sub_path)
            submissions.append((row['Score'], df))
    return submissions


def save_model(model, path='best_model.pkl'):
    """Persist trained model to disk."""
    dump(model, path)


def save_submission(test_df, predictions, path='submission.csv'):
    if isinstance(predictions, pd.Series):
        predictions = predictions.values
    predictions = predictions.astype(int)
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predictions
    })
    submission.to_csv(path, index=False)


def main():
    train_df, test_df = load_data()

    models = {
        "logistic": "Regressão Logística",
        "random_forest": "RandomForest",
        "gradient_boosting": "GradientBoosting",
    }

    # Dados processados uma única vez
    X, y, X_test, _ = preprocess_data(train_df.copy(), test_df.copy(), include_title=True, model_type="logistic")

    scores = {}
    best_estimators = {}

    # GridSearch para o modelo de Regressão Logística
    _, _, _, log_clf = preprocess_data(train_df.copy(), test_df.copy(), include_title=True, model_type="logistic")
    param_grid = {
        'classifier__C': [0.1, 1, 10],
        'classifier__penalty': ['l1', 'l2']
    }
    grid = GridSearchCV(log_clf, param_grid, cv=5)
    grid.fit(X, y)
    print(f"Acurácia {models['logistic']} (GridSearch): {grid.best_score_:.4f}")
    scores['logistic'] = grid.best_score_
    best_estimators['logistic'] = grid.best_estimator_

    # Avalia os demais modelos
    for m in ["random_forest", "gradient_boosting"]:
        _, _, _, clf = preprocess_data(train_df.copy(), test_df.copy(), include_title=True, model_type=m)
        score = cross_val_score(clf, X, y, cv=5).mean()
        print(f"Acurácia {models[m]}: {score:.4f}")
        scores[m] = score
        best_estimators[m] = clf

    best_type = max(scores, key=scores.get)
    print(f"Melhor modelo: {models[best_type]} ({scores[best_type]:.4f})")

    best_clf = best_estimators[best_type]
    predictions, probabilities = train_and_predict(X, y, X_test, best_clf)
    save_model(best_clf)

    submissions = load_previous_submissions()
    if submissions:
        total_weight = scores[best_type]
        weighted = scores[best_type] * probabilities
        for score, df in submissions:
            df = df.set_index('PassengerId').loc[test_df['PassengerId']]
            weighted += score * df['Survived']
            total_weight += score
        final_predictions = (weighted / total_weight >= 0.5).astype(int)
    else:
        final_predictions = predictions

    save_submission(test_df, final_predictions)


if __name__ == '__main__':
    main()
