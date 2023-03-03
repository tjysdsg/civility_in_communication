import pandas as pd
import spacy
from data import load_train_df, load_dev_df, load_demographic_dev_df
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
# noinspection PyUnresolvedReferences
from spacymoji import Emoji


def train():
    # https://github.com/cbaziotis/ekphrasis
    text_processor = TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'number'],
        fix_html=True,  # fix HTML tokens
        # corpus from which the word statistics are going to be used for word segmentation
        segmenter="twitter",
        # corpus from which the word statistics are going to be used for spell correction
        corrector="twitter",
        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=False,  # Unpack contractions (can't -> can not)
        spell_correct_elong=True,  # spell correction for elongated words
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
        dicts=[emoticons],  # emoticons to text
        remove_tags=True,
    )
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe('emoji', first=True)

    def preprocess_text(df: pd.DataFrame):
        text = df['text']

        cleaned_text = []
        f = open('tmp.txt', 'w', encoding='utf-8')
        for t in text:
            t = ' '.join(text_processor.pre_process_doc(t))
            doc = nlp(t)
            t = ' '.join([t.text for t in doc if not t.is_stop and not t.is_punct])
            t = t.replace('url', '')
            t = t.replace('user', '')
            t = t.replace('<', '')
            t = t.replace('>', '')
            t = t.replace('=', '')
            cleaned_text.append(t)

            f.write(f'{cleaned_text[-1]}\n')
        f.close()

        df['text'] = cleaned_text
        return df

    # vectorizer = TfidfVectorizer()
    vectorizer = CountVectorizer()

    train_set = preprocess_text(load_train_df())
    x_train = vectorizer.fit_transform(train_set['text']).toarray()
    y_train = train_set['label']

    dev_set = preprocess_text(load_dev_df())
    x_dev = vectorizer.transform(dev_set['text']).toarray()
    y_dev = dev_set['label']

    model = SGDClassifier(loss="log_loss", penalty="elasticnet")
    # model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_dev)

    print(classification_report(y_dev, y_pred))

    demo_set = load_demographic_dev_df()
    x_demo = vectorizer.transform(demo_set['text']).toarray()
    y_pred = model.predict(x_demo)

    fp = y_pred.sum()
    tn = len(y_pred) - fp
    print('FPR:', fp / (fp + tn))

    """
                      precision    recall  f1-score   support
    
               0       0.78      0.89      0.83       884
               1       0.69      0.50      0.58       440
    
        accuracy                           0.76      1324
       macro avg       0.74      0.70      0.71      1324
    weighted avg       0.75      0.76      0.75      1324
    
    FPR: 0.10173501577287067
    """


def main():
    train()


if __name__ == '__main__':
    main()
