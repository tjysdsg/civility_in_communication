from data import load_dev_df, load_demographic_dev_df
from sklearn.metrics import classification_report, accuracy_score


def main():
    df = load_dev_df()
    df['pred'] = (df['perspective_score'] > 0.8).astype(int)

    print('\nDEV SET')
    print(
        classification_report(df['label'], df['pred'])
    )
    print('Acc:', accuracy_score(df['label'], df['pred']))

    df = load_demographic_dev_df()
    df['label'] = 0
    df['pred'] = (df['perspective_score'] > 0.8).astype(int)

    def fpr(pred):
        fp = pred.sum()
        tn = len(pred) - fp
        return fp / (fp + tn)

    # FPR
    print('\nDEOMO GRUOPS OVERALL')
    print('Acc:', accuracy_score(df['label'], df['pred']))
    print(f'FPR: {fpr(df["pred"])}')

    # Evaluate for each demo group
    print('\nDEOMO GRUOPS')
    demo_groups = ['AA', 'White', 'Hispanic', 'Other']
    for demo in demo_groups:
        d = df[df['demographic'] == demo]
        print(f'Acc for demo group {demo}:', accuracy_score(df['label'], df['pred']))
        print(f'FPR for demo group {demo}: {fpr(d["pred"])}')


# DEV SET
# precision    recall  f1-score   support
#
# 0       0.75      0.98      0.85       884
# 1       0.89      0.33      0.49       440
#
# accuracy                           0.76      1324
# macro avg       0.82      0.66      0.67      1324
# weighted avg       0.79      0.76      0.73      1324
#
# Acc: 0.7643504531722054
#
# DEOMO GRUOPS OVERALL
# Acc: 0.9193611987381703
# FPR: 0.08063880126182965
#
# DEOMO GRUOPS
# Acc for demo group AA: 0.9193611987381703
# FPR for demo group AA: 0.1897590361445783
# Acc for demo group White: 0.9193611987381703
# FPR for demo group White: 0.07319952774498228
# Acc for demo group Hispanic: 0.9193611987381703
# FPR for demo group Hispanic: 0.10149253731343283
# Acc for demo group Other: 0.9193611987381703
# FPR for demo group Other: 0.011764705882352941


if __name__ == '__main__':
    main()
