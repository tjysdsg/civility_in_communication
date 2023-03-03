from data import load_train_df, load_dev_df, load_demographic_dev_df
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


def train():
    tfidf = TfidfVectorizer(min_df=10)

    train_set = load_train_df()
    x_train = tfidf.fit_transform(train_set['text']).toarray()
    y_train = train_set['label']

    dev_set = load_dev_df()
    x_dev = tfidf.transform(dev_set['text']).toarray()
    y_dev = dev_set['label']

    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_dev)

    print(classification_report(y_dev, y_pred))
    """
                      precision    recall  f1-score   support
    
               0       0.76      0.94      0.84       884
               1       0.78      0.39      0.52       440
    
        accuracy                           0.76      1324
       macro avg       0.77      0.67      0.68      1324
    weighted avg       0.76      0.76      0.73      1324
    """

    demo_set = load_demographic_dev_df()
    x_demo = tfidf.transform(demo_set['text']).toarray()
    y_pred = model.predict(x_demo)

    fp = y_pred.sum()
    tn = len(y_pred) - fp
    print('FPR:', fp / (fp + tn))
    # FPR: 0.11139589905362776


def main():
    train()


if __name__ == '__main__':
    main()
