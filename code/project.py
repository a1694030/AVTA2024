import pandas as pd
import numpy as np
from tqdm import tqdm

#读取数据
df = pd.read_csv('data/app_use_info_label.csv', sep=',', header=0,encoding='gbk')
df2 = pd.read_csv('data/user_portrait.csv', sep=',', header=0)
df3 = pd.read_csv('data/user_trajectory2.csv', sep=',', header=0).drop(columns='Unnamed: 0')
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
    df_copy[prefix + 'dayofweek'] = df_copy[col].dt.dayofweek
    df_copy[prefix + 'is_wknd'] = df_copy[col].dt.dayofweek // 6
    df_copy[prefix + 'quarter'] = df_copy[col].dt.quarter
    df_copy[prefix + 'is_month_start'] = df_copy[col].dt.is_month_start.astype(int)
    df_copy[prefix + 'is_month_end'] = df_copy[col].dt.is_month_end.astype(int)
    del df_copy[col]
    
    return df_copy   
    
df_all = get_time_feature(df_all, 'stime')
df_all = get_time_feature(df_all, 'end_time')

#label编码
from sklearn import preprocessing
 
enc=preprocessing.LabelEncoder() 
enc=enc.fit(df_all['app_class_1']) 
df_all['app_class_1']=enc.transform(df_all['app_class_1'])

enc2=preprocessing.LabelEncoder() 
enc2=enc2.fit(df_all['app_class_2']) 
df_all['app_class_2']=enc2.transform(df_all['app_class_2'])

#特征衍生
df_all['up_flow']= df_all['up_flow']/1024
df_all['down_flow']= df_all['down_flow']/1024

df_all['app1_upflow_mean'] = df_all['app_class_1'].map(df_all.groupby('app_class_1')['up_flow'].mean())
df_all['app2_upflow_mean'] = df_all['app_class_2'].map(df_all.groupby('app_class_2')['up_flow'].mean())
df_all['app1_upflow_max'] = df_all['app_class_1'].map(df_all.groupby('app_class_1')['up_flow'].max())
df_all['app2_upflow_max'] = df_all['app_class_2'].map(df_all.groupby('app_class_2')['up_flow'].max())
df_all['app1_upflow_min'] = df_all['app_class_1'].map(df_all.groupby('app_class_1')['up_flow'].min())
df_all['app2_upflow_min'] = df_all['app_class_2'].map(df_all.groupby('app_class_2')['up_flow'].min())

df_all['app1_downflow_mean'] = df_all['app_class_1'].map(df_all.groupby('app_class_1')['down_flow'].mean())
df_all['app2_downflow_mean'] = df_all['app_class_2'].map(df_all.groupby('app_class_2')['down_flow'].mean())
df_all['app1_downflow_max'] = df_all['app_class_1'].map(df_all.groupby('app_class_1')['down_flow'].max())
df_all['app2_downflow_max'] = df_all['app_class_2'].map(df_all.groupby('app_class_2')['down_flow'].max())
df_all['app1_downflow_min'] = df_all['app_class_1'].map(df_all.groupby('app_class_1')['down_flow'].min())
df_all['app2_downflow_min'] = df_all['app_class_2'].map(df_all.groupby('app_class_2')['down_flow'].min())

df_all['app1_consume_mean'] = df_all['app_class_1'].map(df_all.groupby('app_class_1')['consume'].mean())
df_all['app2_consume_mean'] = df_all['app_class_2'].map(df_all.groupby('app_class_2')['consume'].mean())
df_all['app1_consume_max'] = df_all['app_class_1'].map(df_all.groupby('app_class_1')['consume'].max())
df_all['app2_consume_max'] = df_all['app_class_2'].map(df_all.groupby('app_class_2')['consume'].max())
df_all['app1_consume_min'] = df_all['app_class_1'].map(df_all.groupby('app_class_1')['consume'].min())
df_all['app2_consume_min'] = df_all['app_class_2'].map(df_all.groupby('app_class_2')['consume'].min())

df_all['app1_age_mean'] = df_all['app_class_1'].map(df_all.groupby('app_class_1')['age'].mean())
df_all['app2_age_mean'] = df_all['app_class_2'].map(df_all.groupby('app_class_2')['age'].mean())
df_all['app1_age_max'] = df_all['app_class_1'].map(df_all.groupby('app_class_1')['age'].max())
df_all['app2_age_max'] = df_all['app_class_2'].map(df_all.groupby('app_class_2')['age'].max())
df_all['app1_age_min'] = df_all['app_class_1'].map(df_all.groupby('app_class_1')['age'].min())
df_all['app2_age_min'] = df_all['app_class_2'].map(df_all.groupby('app_class_2')['age'].min())

df_all['up-down'] = df_all['up_flow']-df_all['down_flow']
df_all['up/down'] = df_all['up_flow']/df_all['down_flow']
df_all['up+down'] = df_all['up_flow']+df_all['down_flow']

#分离训练集，测试集
from sklearn.model_selection import train_test_split

df_all=df_all.replace([np.inf, -np.inf], 0)
df_all.fillna(0,inplace=True)
train_df,test_df = train_test_split(df_all,train_size=0.8,shuffle=True,random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
feature_cols = [cols for cols in df_all if cols not in ['msisdn','label','end_time','stime','times_month']]
len(feature_cols)

#构建模型
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier 
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import roc_auc_score,accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from pytorch_tabnet import tab_model
from sklearn.svm import SVC
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
import torch
import warnings
warnings.filterwarnings('ignore')

# 标准化
def scale(train_features,test_features):
    scaler=StandardScaler()
    scaler.fit(train_features)
    train_features=pd.DataFrame(scaler.transform(train_features),columns=test_features.keys())
    test_features=pd.DataFrame(scaler.transform(test_features),columns=test_features.keys())
    return train_features,test_features

# 模型定义
def ml_model(clf,train_x,train_y,test_x=[],test_y=[]):
    seeds=[888]
    train_oof = np.zeros([train_x.shape[0],3])
    feat_imp_df = pd.DataFrame()
    feat_imp_df['feature'] = train_x.columns
    feat_imp_df['imp'] = 0
    #标准化
    scaler=StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    if (len(test_x)!=0) and (len(test_x)!=0):
        test_oof = np.zeros([test_x.shape[0],3])
        test_x = scaler.transform(test_x)
    for seed in seeds:
        print('Seed:',seed)
        folds = 5
        kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        acc_scores_val = []
        acc_scores_test = []
        for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
            trn_x, trn_y, val_x, val_y = train_x[train_index], train_y[train_index], train_x[valid_index], \
                                        train_y[valid_index] 
            if clf == 'xgb':
                print("|  XGB  Fold  {}  Training Start           |".format(str(i + 1)))
                xgb_params = {'booster': 'gbtree','objective': 'multi:softprob','eval_metric':'mlogloss','num_class':3,
                    'n_estimators':500,'max_depth': 8,'lambda': 10,'subsample': 0.7,'colsample_bytree': 0.8,'eta': 0.1,
                    'colsample_bylevel': 0.7,'tree_method': 'hist','seed': seed,'nthread': 16}
                #训练模型
                model = xgb.XGBClassifier(*xgb_params)
                model.fit(trn_x,trn_y,eval_set=[(trn_x, trn_y),(val_x,val_y)],early_stopping_rounds=50,verbose=100)
                #验证集
                print('************ Val_Result ************')
                val_pred  = model.predict_proba(val_x)
                acc_score_val = accuracy_score(val_y, np.argmax(val_pred, axis=1))
                acc_scores_val.append(acc_score_val)
                print('AVG_acc :',sum(acc_scores_val)/len(acc_scores_val))
                print('XGB_result :',classification_report(val_y, np.argmax(val_pred, axis=1)))
                #测试集
                if (len(test_x)!=0) and (len(test_x)!=0):
                    print('************ Test_Result ************')
                    test_pred  = model.predict_proba(test_x)
                    acc_score_tset = accuracy_score(test_y, np.argmax(test_pred, axis=1))
                    acc_scores_test.append(acc_score_tset)
                    print('AVG_acc :',sum(acc_scores_test)/len(acc_scores_test))
                    print('XGB_result :',classification_report(test_y, np.argmax(test_pred, axis=1)))
                    #保存测试集结果
                    test_oof += test_pred / kf.n_splits / len(seeds)
                
                #保存训练集结果
                train_oof[valid_index] = val_pred / kf.n_splits / len(seeds)
                
                #模型特征重要性
                feat_imp_df['imp'] += model.feature_importances_ / folds/ len(seeds)
                feat_imp_df = feat_imp_df.sort_values(by='imp', ascending=False).reset_index(drop=True)
                feat_imp_df['rank'] = range(feat_imp_df.shape[0])
            if clf == 'lgb':
                lgb_params = {'boosting_type': 'gbdt','n_estimators':500,'min_child_weight': 4,'num_leaves': 64,
                    'feature_fraction': 0.8,'bagging_fraction': 0.8,'bagging_freq': 4,'learning_rate': 0.02,
                    'seed': seed,'nthread': 32,'n_jobs':8,'verbose': -1}
                print("|  LGB  Fold  {}  Training Start           |".format(str(i + 1)))
                #训练模型
                model = lgb.LGBMClassifier(**lgb_params)
                model.fit(trn_x,trn_y)
                
                #验证集
                print('************ Val_Result ************')
                val_pred  = model.predict_proba(val_x)
                acc_score_val = accuracy_score(val_y, np.argmax(val_pred, axis=1))
                acc_scores_val.append(acc_score_val)
                print('AVG_acc :',sum(acc_scores_val)/len(acc_scores_val))
                print('Cat_result :',classification_report(val_y, np.argmax(val_pred, axis=1)))
                #测试集
                if (len(test_x)!=0) and (len(test_x)!=0):
                    print('************ Test_Result ************')
                    test_pred  = model.predict_proba(test_x)
                    acc_score_tset = accuracy_score(test_y, np.argmax(test_pred, axis=1))
                    acc_scores_test.append(acc_score_tset)
                    print('AVG_acc :',sum(acc_scores_test)/len(acc_scores_test))
                    print('Cat_result :',classification_report(test_y, np.argmax(test_pred, axis=1))) 
                    #保存测试集结果
                    test_oof += test_pred / kf.n_splits / len(seeds)
                #保存训练集结果
                train_oof[valid_index] = val_pred / kf.n_splits / len(seeds)
            if clf == 'cat':
                print("|  cat  Fold  {}  Training Start           |".format(str(i + 1)))
                #训练模型
                model = CatBoostClassifier(verbose=False)
                model.fit(trn_x,trn_y)
                    
                #验证集
                print('************ Val_Result ************')
                val_pred  = model.predict_proba(val_x)
                acc_score_val = accuracy_score(val_y, np.argmax(val_pred, axis=1))
                acc_scores_val.append(acc_score_val)
                print('AVG_acc :',sum(acc_scores_val)/len(acc_scores_val))
                print('Cat_result :',classification_report(val_y, np.argmax(val_pred, axis=1)))
                #测试集
                if (len(test_x)!=0) and (len(test_x)!=0):
                    print('************ Test_Result ************')
                    test_pred  = model.predict_proba(test_x)
                    acc_score_tset = accuracy_score(test_y, np.argmax(test_pred, axis=1))
                    acc_scores_test.append(acc_score_tset)
                    print('AVG_acc :',sum(acc_scores_test)/len(acc_scores_test))
                    print('Cat_result :',classification_report(test_y, np.argmax(test_pred, axis=1))) 
                    #保存测试集结果
                    test_oof += test_pred / kf.n_splits / len(seeds)
                    
                #保存结果
                train_oof[valid_index] = val_pred / kf.n_splits / len(seeds)
            if clf == 'svm':
                print("|  SVM  Fold  {}  Training Start           |".format(str(i + 1)))
                #训练模型
                model = SVC(kernel='rbf', C=1, gamma='auto', probability=True,max_iter=1000)
                model.fit(trn_x,trn_y)
                
                #验证集
                print('************ Val_Result ************')
                val_pred  = model.predict_proba(val_x)
                acc_score_val = accuracy_score(val_y, np.argmax(val_pred, axis=1))
                acc_scores_val.append(acc_score_val)
                print('AVG_acc :',sum(acc_scores_val)/len(acc_scores_val))
                print('Svm_result :',classification_report(val_y, np.argmax(val_pred, axis=1)))
                #测试集
                print('************ Test_Result ************')
                test_pred  = model.predict_proba(test_x)
                acc_score_tset = accuracy_score(test_y, np.argmax(test_pred, axis=1))
                acc_scores_test.append(acc_score_tset)
                print('AVG_acc :',sum(acc_scores_test)/len(acc_scores_test))
                print('Svm_result :',classification_report(test_y, np.argmax(test_pred, axis=1))) 
                #保存结果
                train_oof[valid_index] = val_pred / kf.n_splits / len(seeds)
                test_oof += test_pred / kf.n_splits / len(seeds)
                
            if clf == 'tabnet':
                print(f"     Tab_model  Fold {i+1}  Training Starting       ")
                if torch.cuda.is_available():
                    print("Using GPU")
                    device = "cuda"
                else:
                    print("Using CPU")
                    device = "cpu"
                    
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = tab_model.TabNetClassifier()

                model.fit(trn_x, trn_y,eval_set=[(val_x, val_y)],eval_metric=['accuracy'])
                #验证集
                print('************ Val_Result ************')
                val_pred  = model.predict_proba(val_x)
                acc_score_val = accuracy_score(val_y, np.argmax(val_pred, axis=1))
                acc_scores_val.append(acc_score_val)
                print('AVG_acc :',sum(acc_scores_val)/len(acc_scores_val))
                print('Tabnet_result :',classification_report(val_y, np.argmax(val_pred, axis=1)))
                #测试集
                print('************ Test_Result ************')
                test_pred  = model.predict_proba(test_x)
                acc_score_tset = accuracy_score(test_y, np.argmax(test_pred, axis=1))
                acc_scores_test.append(acc_score_tset)
                print('AVG_acc :',sum(acc_scores_test)/len(acc_scores_test))
                print('Tabnet_result :',classification_report(test_y, np.argmax(test_pred, axis=1))) 
                #保存结果
                train_oof[valid_index] = val_pred / kf.n_splits / len(seeds)
                test_oof += test_pred / kf.n_splits / len(seeds)
        if (len(test_x)!=0) and (len(test_x)!=0):
            return train_oof,test_oof,model
        else:
            return train_oof,model

# 训练 XGB模型
xgb_train_oof,xgb_test_oof, xgb_model = ml_model('xgb',train_df[feature_cols], train_df['label'],test_df[feature_cols], test_df['label'])

# 训练 LGB模型
lgb_train_oof,lgb_test_oof,lgb_model = ml_model('lgb',train_df[feature_cols], train_df['label'],test_df[feature_cols], test_df['label'])

# 训练 CAT模型
cat_train_oof,cat_test_oof,cat_model = ml_model('cat',train_df[feature_cols], train_df['label'],test_df[feature_cols], test_df['label'])

# # 训练 SVM模型
# svm_oof,svm_model = ml_model('svm',train_df[feature_cols], train_df['label'],test_df[feature_cols], test_df['label'])

# # 训练 Tabnet模型
# tab_oof,tabnet_model = ml_model('tabnet',train_df[feature_cols], train_df['label'],test_df[feature_cols], test_df['label'])

def hard_sample(xgb_oof,lgb_oof,cat_oof,df):
    df_pre = pd.DataFrame()
    df_pre['xgb_pre'] = np.argmax(xgb_oof,axis=1)
    df_pre['lgb_pre'] = np.argmax(lgb_oof,axis=1)
    df_pre['cat_pre'] = np.argmax(cat_oof,axis=1)
    df_pre['label'] = df['label']

    grade_list = []
    for row in df_pre.itertuples():
        grade = 0
        if getattr(row,'xgb_pre') == getattr(row,'label'):
            grade += 1
        if getattr(row,'lgb_pre') == getattr(row,'label'):
            grade += 1
        if getattr(row,'cat_pre') == getattr(row,'label'):
            grade += 1
        grade_list.append(grade)

    #困难样本
    df_pre['grade'] = grade_list
    hard_index = df_pre.loc[(df_pre['grade']==0)].index

    # 困难样本独立训练
    # 训练 XGB模型
    hard_df = df.loc[hard_index].reset_index(drop=True)
    xgb_oof_2, xgb_model_2 = ml_model('xgb',hard_df[feature_cols],hard_df['label'])
    # 训练 LGB模型
    lgb_oof_2,lgb_model_2 = ml_model('lgb',hard_df[feature_cols], hard_df['label'])

    # 训练 cat模型
    cat_oof_2,cat_model_2 = ml_model('cat',hard_df[feature_cols], hard_df['label'])

    #替换困难样本结果
    xgb_oof = np.argmax(xgb_oof,axis=1)
    xgb_oof[hard_index]=np.argmax(xgb_oof_2,axis=1)

    lgb_oof = np.argmax(lgb_oof,axis=1)
    lgb_oof[hard_index]=np.argmax(lgb_oof_2,axis=1)

    cat_oof = np.argmax(cat_oof,axis=1)
    cat_oof[hard_index]=np.argmax(cat_oof_2,axis=1)
    
    return xgb_oof,lgb_oof,cat_oof

xgb_train_oof,lgb_train_oof,cat_train_oof = hard_sample(xgb_train_oof,lgb_train_oof,cat_train_oof,train_df)
# xgb_test_oof,lgb_test_oof,cat_test_oof = hard_sample(xgb_test_oof,lgb_test_oof,cat_test_oof,test_df)

#xgb预测结果作为新特征(替换困难样本)
train_df['xgb_pre'] = xgb_train_oof
test_df['xgb_pre'] = xgb_test_oof
#lgb预测结果作为新特征(替换困难样本)
train_df['lgb_pre'] = lgb_train_oof
test_df['lgb_pre'] = lgb_test_oof
#cat预测结果作为新特征(替换困难样本)
train_df['cat_pre'] = cat_train_oof
test_df['cat_pre'] = cat_test_oof

#xgb预测结果作为新特征
# df_all['xgb_pre'] = np.argmax(xgb_oof,axis=1)
# #lgb预测结果作为新特征
# df_all['lgb_pre'] = np.argmax(lgb_oof,axis=1)
# #cat预测结果作为新特征
# df_all['cat_pre'] = np.argmax(cat_oof,axis=1)

#训练特征
feature_cols = [cols for cols in train_df if cols not in ['msisdn','label','end_time','stime','times_month']]
# 训练tabnet模型
final_tab_train_oof,final_tab_test_oof,final_tab_model = ml_model('tabnet',train_df[feature_cols], train_df['label'],test_df[feature_cols], test_df['label'])

import joblib
#模型保存
def save_model(mdoel,save_path):
    # save_path = save_path + '/' +  mdoel_name +'.pkl'
    joblib.dump(mdoel, save_path)
#模型保存路径
xgb_model_path = '../model/xgb.pkl'
lgb_model_path = '../model/lgb.pkl'
cat_model_path = '../model/cat.pkl'
svm_model_path = '../model/svm.pkl'
tabnet_model_path = '../model/tabnet.pkl'
final_tab_model_path = '../model/final_tab.pkl'
#保存
save_model(xgb_model,xgb_model_path)
save_model(lgb_model, lgb_model_path)
save_model(cat_model,  cat_model_path)
save_model(svm_model,svm_model_path)
save_model(tabnet_model, tabnet_model_path)
save_model(final_tab_model,  final_tab_model_path)







