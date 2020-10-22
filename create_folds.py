import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
mskf = StratifiedKFold(n_splits=5, random_state=69420, shuffle=True)

train_df2 = pd.read_csv('data/train.csv')
# train_df2 = train_df2.sample(frac=1).reset_index(drop=True)
train_df2 = train_df2.drop(['gleason_score'], axis=1)

# Remove blank image
train_df2 = train_df2[train_df2['image_id'] !=
                      '3790f55cad63053e956fb73027179707'].reset_index(drop=True)

X, y = train_df2.values[:, 0:2], train_df2[['isup_grade']].values[:, 0]

train_df2['fold'] = -1
for fld, (_, test_idx) in enumerate(mskf.split(X, y)):
    train_df2.iloc[test_idx, -1] = fld

train_df2.to_csv('data/train_with_folds.csv')
