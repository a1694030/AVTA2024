import pandas as pd
import numpy as np
from tqdm import tqdm

#读取数据
df = pd.read_csv(r'C:\Users\86130\Desktop\ML算法竞赛\梧桐杯\省决赛\data\app_use_info.csv', sep=',', header=0)
df2 = pd.read_csv(r'C:\Users\86130\Desktop\ML算法竞赛\梧桐杯\省决赛\data\user_portrait.csv', sep=',', header=0)
df3 = pd.read_csv(r'C:\Users\86130\Desktop\ML算法竞赛\梧桐杯\省决赛\data\user_trajectory2.csv', sep=',', header=0)

df['潜在用户评分'] = np.zeros(len(df))
# #根据支'导航','团购','电商','旅游'类APP月使用次数进行潜客标记，绝对值大于3*mean设1，反之设0

for i in ['导航','团购','电商','旅游','出行','股票','信贷','企业','理财','支付']:
        mean = df[df['app_class_1'] == i]['times_month'].mean()
        std = df[df['app_class_1'] == i]['times_month'].std()
        df.loc[(df['app_class_1'] == i) & (df['times_month'] < mean-(3*std)), '潜在用户评分'] = 0
        # df.loc[(df['app_class_1'] == i) & (mean-(3*std)<df['times_month']) & (df['times_month']<mean-(2*std)), '潜在用户评分'] = 1
        df.loc[(df['app_class_1'] == i) & (mean-(2*std)<df['times_month']) & (df['times_month']<mean-std), '潜在用户评分'] = 1
        df.loc[(df['app_class_1'] == i) & (mean-(std)<df['times_month'])&(df['times_month']<mean), '潜在用户评分'] = 2
        
        df.loc[(df['app_class_1'] == i) & (mean+(std)>df['times_month'])&(df['times_month']>mean), '潜在用户评分'] = 3
        df.loc[(df['app_class_1'] == i) & (mean+(2*std)>df['times_month']) & (df['times_month']>mean+std), '潜在用户评分'] = 4
        df.loc[(df['app_class_1'] == i) & (mean+(3*std)>df['times_month']) & (df['times_month']>mean+(2*std)), '潜在用户评分'] = 5
        df.loc[(df['app_class_1'] == i) & (df['times_month'] > mean + (3*std)), '潜在用户评分'] = 6

#合并数据
df_all = pd.merge(df, df2, left_on='msisdn', right_on='userid', how='left').drop(columns = ['userid'])
df_all = df_all.merge(df3, on = 'msisdn', how = 'left')

#日期特征处理
def get_time_feature(df, col):
    
    df_copy = df.copy()
    prefix = col + "_"
    df_copy['new_'+col] = df_copy[col].astype(str)
    
    col = 'new_'+col
    df_copy[col] = pd.to_datetime(df_copy[col])
    df_copy[prefix + 'year'] = df_copy[col].dt.year
    df_copy[prefix + 'month'] = df_copy[col].dt.month
    df_copy[prefix + 'day'] = df_copy[col].dt.day
    # df_copy[prefix + 'weekofyear'] = df_copy[col].dt.weekofyear
    df_copy[prefix + 'dayofweek'] = df_copy[col].dt.dayofweek
    df_copy[prefix + 'is_wknd'] = df_copy[col].dt.dayofweek // 6
    df_copy[prefix + 'quarter'] = df_copy[col].dt.quarter
    df_copy[prefix + 'is_month_start'] = df_copy[col].dt.is_month_start.astype(int)
    df_copy[prefix + 'is_month_end'] = df_copy[col].dt.is_month_end.astype(int)
    del df_copy[col]
    
    return df_copy   
    
df_all = get_time_feature(df_all, 'stime')
df_all = get_time_feature(df_all, 'end_time')

#目标编码
cat_cols = ['times_month','age']
num_cols = ['up_flow','down_flow','region','consume']

df_all['app_class_1'] = df_all['app_class_1'].map(df_all.groupby(['app_class_1'])['潜在用户评分'].mean())
df_all['app_class_2'] = df_all['app_class_2'].map(df_all.groupby(['app_class_2'])['潜在用户评分'].mean())
for i in cat_cols :
    df_all[f'{i}_target_mean'] = df_all[i].map(df_all.groupby([i])['潜在用户评分'].mean())
    df_all[f'{i}_count'] = df_all[i].map(df_all[i].value_counts())
for i in cat_cols :
    for j in num_cols:
        df_all[f'{i}_{j}_mean'] = df_all[i].map(df_all.groupby([i])[j].mean())
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans

group_cluster_cols = {
    'group1': ['up_flow', 'down_flow','up_flow'],
    'group2': ['down_flow','region','consume'],
    'group2': ['app_class_1','app_class_2','consume'],
}

for group, cols in tqdm(group_cluster_cols.items()):
    mbk = MiniBatchKMeans(
        init="k-means++",
        n_clusters=50,
        batch_size=2048,
        n_init=10,
        max_no_improvement=10,
        verbose=0,
        random_state=512
    )
    X = MinMaxScaler().fit_transform(df_all[cols].values)
    df_all[f'{group}_cluster'] = mbk.fit_predict(X)

#等宽分箱
num_bins = 20
cut_labels = [i for i in range(num_bins)]
for col in tqdm(['up_flow','down_flow']):
    df_all[f'{col}_bin'] = pd.cut(df_all[col],num_bins,labels=cut_labels).apply(int)

#分离训练集，测试集
df_all=df_all.replace([np.inf, -np.inf], 0)
df_all.fillna(0,inplace=True)
drop_cols = ['end_time_quarter','end_time_is_month_start','end_time_is_month_end','up_flow_bin','down_flow_bin']
feature_cols = [cols for cols in df_all if cols not in ['msisdn','潜在用户评分','end_time','stime','times_month']+drop_cols]
len(feature_cols)

#构建模型
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import roc_auc_score,accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pytorch_tabnet import tab_model
from sklearn.svm import SVC
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
import torch
import warnings
warnings.filterwarnings('ignore')

def ml_model(clf,train_x, train_y):
    seeds=[42]
    oof = np.zeros([train_x.shape[0],7])
    feat_imp_df = pd.DataFrame()
    feat_imp_df['feature'] = train_x.columns
    feat_imp_df['imp'] = 0
    for seed in seeds:
        print('Seed:',seed)
        folds = 5
        kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        acc_scores = []
        # train_x = train_x.values
        # train_y = train_y.values
        if clf == xgb:
            for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
                trn_x, trn_y, val_x, val_y = train_x.values[train_index], train_y.values[train_index], train_x.values[valid_index], \
                                            train_y.values[valid_index] 
                print("|  XGB  Fold  {}  Training Start           |".format(str(i + 1)))
                xgb_params = {
                    'booster': 'gbtree',
                    'objective': 'multi:softprob',
                    'eval_metric':'mlogloss',
                    'num_class':7,
                    'n_estimators':100,
                    'max_depth': 8,
                    'lambda': 10,
                    'subsample': 0.7,
                    'colsample_bytree': 0.8,
                    'colsample_bylevel': 0.7,
                    'eta': 0.1,
                    'tree_method': 'hist',
                    'seed': seed,
                    'nthread': 16
                }
                
                #训练模型
                xgb_model = clf.XGBClassifier(*xgb_params)
                xgb_model.fit(trn_x,trn_y,eval_set=[(trn_x, trn_y),(val_x,val_y)],early_stopping_rounds=50,verbose=100)
                
                val_pred  = xgb_model.predict_proba(val_x)
                feat_imp_df['imp'] += xgb_model.feature_importances_ / folds/ len(seeds)
                feat_imp_df = feat_imp_df.sort_values(by='imp', ascending=False).reset_index(drop=True)
                feat_imp_df['rank'] = range(feat_imp_df.shape[0])
                
                oof[valid_index] = val_pred / kf.n_splits / len(seeds)
                
                acc_score = accuracy_score(val_y, np.argmax(val_pred, axis=1))
                acc_scores.append(acc_score)
                print('AVG_acc :',sum(acc_scores)/len(acc_scores))
                print('XGB :',classification_report(val_y, np.argmax(val_pred, axis=1)))
            return oof,feat_imp_df,xgb_model
        if clf == tab_model:
            for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
                trn_x, trn_y, val_x, val_y = train_x.values[train_index], train_y.values[train_index], train_x.values[valid_index], \
                                            train_y.values[valid_index] 
                print("     Tab_model  Fold   Training Starting       ")
                if torch.cuda.is_available():
                    print("Using GPU")
                    device = "cuda"
                else:
                    print("Using CPU")
                    device = "cpu"

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(trn_x)
                X_val = scaler.fit_transform(val_x)
                # X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, train_y, test_size=0.2, random_state=42)
                torch.manual_seed(seed)
                np.random.seed(seed)
                tabnet = tab_model.TabNetClassifier(
                    seed=seed,
                    verbose=1,
                    device_name=device,  # Use the available device (GPU or CPU)
                )

                tabnet.fit(
                        X_scaled, trn_y,
                        eval_set=[(X_val, val_y)],
                        eval_metric=['accuracy'],
                        max_epochs=50,
                        patience=5,
                        batch_size=512,  #
                    )
                oof[valid_index] += tabnet.predict_proba(X_val)/ kf.n_splits
            return oof,feat_imp_df,tabnet                  


# 训练 XGB模型
xgb_oof, xgb_imp_df,xgb_model = ml_model(xgb,df_all[feature_cols], df_all['潜在用户评分'])

#xgb预测结果作为新特征
df_all['pre'] = np.argmax(xgb_oof,axis=1)
# 训练tabnet模型
tab_oof, tab_imp_df,tabnet_model = ml_model(tab_model,df_all[feature_cols], df_all['潜在用户评分'])

print('AVG_acc :', accuracy_score(df_all['潜在用户评分'], np.argmax(tab_oof,axis=1)))
print('XGB :',classification_report(df_all['潜在用户评分'], np.argmax(tab_oof,axis=1)))

