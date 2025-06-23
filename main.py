# Simple baseline model for Kaggle Titanic competition
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer


def load_data():
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def preprocess_data(train_df, test_df):
    y = train_df['Survived']
    X = train_df.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
    X_test = test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
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
    return predictions


def save_submission(test_df, predictions, path='submission.csv'):
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predictions
    })
    submission.to_csv(path, index=False)


def main():
    train_df, test_df = load_data()
    X, y, X_test, clf = preprocess_data(train_df, test_df)
    predictions = train_and_predict(X, y, X_test, clf)
    save_submission(test_df, predictions)


if __name__ == '__main__':
    main()
