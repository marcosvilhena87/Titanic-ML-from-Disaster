# Simple baseline model for Kaggle Titanic competition
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score


def load_data():
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def preprocess_data(train_df, test_df):
    y = train_df['Survived']

    for df in (train_df, test_df):
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

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

    clf = Pipeline(
        steps=[('preprocessor', preprocessor),
               ('classifier', LogisticRegression(max_iter=1000))]
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
    X, y, X_test, clf = preprocess_data(train_df, test_df)

    # estimate baseline performance via cross validation
    baseline_score = cross_val_score(clf, X, y, cv=5).mean()

    predictions, probabilities = train_and_predict(X, y, X_test, clf)

    submissions = load_previous_submissions()
    if submissions:
        total_weight = baseline_score
        weighted = baseline_score * probabilities
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
