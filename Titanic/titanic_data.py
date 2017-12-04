import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing

### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    # y即目标年龄
    y = known_age[:, 0]
    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges

    return df, rfr

def set_missing_ages_in_test(df,rfr):
    tmp_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    null_age = tmp_df[df.Age.isnull()].as_matrix()
    # 根据特征属性X预测年龄并补上
    X = null_age[:, 1:]
    predictedAges = rfr.predict(X)
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges
    return df

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

def set_dummies(df):
    dummies_Cabin = pd.get_dummies(df['Cabin'], prefix='Cabin')

    dummies_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')

    dummies_Sex = pd.get_dummies(df['Sex'], prefix='Sex')

    dummies_Pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')

    df = pd.concat([df, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

    return df

def set_regularize(df):
    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(df['Age'])
    df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
    fare_scale_param = scaler.fit(df['Fare'])
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
    return df

def trainset_process(df):
    data_train, rfr = set_missing_ages(df)

    data_train = set_Cabin_type(data_train)

    data_train = set_dummies(data_train)

    data_train = set_regularize(data_train)
    return data_train,rfr

def testset_process(df,rfr):
    df.loc[(df.Fare.isnull()), 'Fare'] = 0

    data_test = set_missing_ages_in_test(df,rfr)

    data_test = set_Cabin_type(data_test)

    data_test = set_dummies(data_test)

    data_test = set_regularize(data_test)
    return data_test

# class DataSets:
#     test_data = None
#     val_data = None
#     train_data = None
#
#     def __init__(self):
#         pass
#
#     def get_train_x(self):
#
#     def get_train_y(self):
#
#     def get_test_x(self):
#
#     def get_test_y(self):

def read_data_sets(data_dir):
    if data_dir is None:
        return "the dir is none"

    data_train = pd.read_csv(data_dir+"/train.csv")
    data_test = pd.read_csv(data_dir+"/test.csv")

    # data_train.info()
    # data_test.info()

    data_train,rfr = trainset_process(data_train)
    data_test = testset_process(data_test,rfr)

    # print(data_train)

    data_train.to_csv(data_dir+"/processedtrain.csv",index=False)
    data_test.to_csv(data_dir+"/processedtest.csv",index=False)

    print("data process finished")

    return data_train,data_test

def read_data_sets_from_csv(data_dir):
    data_train = pd.read_csv(data_dir + "/processedtrain.csv")
    data_test = pd.read_csv(data_dir + "/processedtest.csv")
    return data_train, data_test


data_tain,data_test = read_data_sets("./data")


