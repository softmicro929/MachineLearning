import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
import re
import matplotlib.pyplot as plt

### 使用 RandomForestClassifier 填补缺失的年龄属性
from Titanic.DataSets import DataSets


def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
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
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr


def set_missing_ages_in_test(df, rfr):
    tmp_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    null_age = tmp_df[df.Age.isnull()].as_matrix()
    # 根据特征属性X预测年龄并补上
    X = null_age[:, 1:]
    predictedAges = rfr.predict(X)
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges
    return df


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


def set_Cabin_Deck(df):
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'U0'
    # create feature for the alphabetical part of the cabin number
    df['CabinLetter'] = df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    # factorize函数可以将Series中的标称型数据映射称为一组数字，相同的标称型映射为相同的数字。
    # factorize函数的返回值是一个tuple（元组），元组中包含两个元素。
    # 第一个元素是一个array，其中的元素是标称型元素映射为的数字；
    # 第二个元素是Index类型，其中的元素是所有标称型元素，没有重复。
    df['CabinLetter'] = pd.factorize(df['CabinLetter'])[0]
    return df


def set_dummies(df):
    # dummies_Cabin = pd.get_dummies(df['Cabin'], prefix='Cabin')
    # dummies_Cabin = pd.get_dummies(df['CabinLetter'], prefix='CabinLetter')

    dummies_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')

    dummies_Sex = pd.get_dummies(df['Sex'], prefix='Sex')

    dummies_Pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')

    df = pd.concat([df, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

    return df


def set_regularize(df):
    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(df['Age'])
    df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
    fare_scale_param = scaler.fit(df['Fare'])
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
    return df


# 将票价分段，转成离散值
def fare_factorize(df):
    df['Fare_bin'] = pd.qcut(df['Fare'], 4)
    # qcut() creates a new variable that identifies the quartile range, but we can't use the string so either
    # factorize or create dummies from the result
    # df['Fare_bin_id'] = pd.factorize(df['Fare_bin'])[0]
    df = pd.concat([df, pd.get_dummies(df['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))], axis=1)
    return df


def title_extra_inname(df):
    df['Surname'] = df['Name'].map(lambda x: re.compile(
        "(Mr|Mrs|Miss|Master|Don|Rev|Dr|Mme|Ms|Major|Lady|Sir|Mlle|Col|Capt|the Countess|Jonkheer|Dona)\.\s(\w*)").findall(
        x)[0][1])
    df['Surname'] = pd.factorize(df['Surname'])[0]
    # 处理称谓
    df['Title'] = df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
    # print(df['Title'])
    df['Title'].loc[df.Title == 'Jonkheer'] = 'Master'
    df['Title'].loc[df.Title.isin(['Ms', 'Mlle'])] = 'Miss'
    df['Title'].loc[df.Title == 'Mme'] = 'Mrs'
    df['Title'].loc[df.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
    df['Title'].loc[df.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'
    df['Title_id'] = pd.factorize(df['Title'])[0] + 1
    # df.drop(['Name'], axis=1, inplace=True)
    # df.drop(['Names'], axis=1, inplace=True)
    return df


def feature_drop(df):
    df.drop(['Name', 'Surname', 'Title', 'Fare_bin', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1,
            inplace=True)
    return df


def trainset_process(df):
    data_train, rfr = set_missing_ages(df)

    # 这里将cabin的甲板信息加进来
    # data_train = set_Cabin_type(data_train)
    data_train = set_Cabin_Deck(data_train)

    data_train = fare_factorize(data_train)

    data_train = set_dummies(data_train)

    data_train = set_regularize(data_train)

    data_train = title_extra_inname(data_train)

    data_train = feature_drop(data_train)

    return data_train, rfr


def testset_process(df, rfr):
    df.loc[(df.Fare.isnull()), 'Fare'] = 0

    data_test = set_missing_ages_in_test(df, rfr)

    # data_test = set_Cabin_type(data_test)
    data_test = set_Cabin_Deck(data_test)

    data_test = fare_factorize(data_test)

    data_test = set_dummies(data_test)

    data_test = set_regularize(data_test)

    data_test = title_extra_inname(data_test)

    data_test = feature_drop(data_test)

    return data_test


def raw_data_process(data_dir):
    if data_dir is None:
        return "the dir is none"

    data_train = pd.read_csv(data_dir + "/train.csv")
    data_test = pd.read_csv(data_dir + "/test.csv")

    # data_train.info()
    # data_test.info()

    data_train, rfr = trainset_process(data_train)
    data_test = testset_process(data_test, rfr)

    # 从train——data中分出300个做validation
    data_train = data_train.filter(
        regex='Survived|Age_.*|SibSp|Parch|Fare_.*|CabinLetter_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_id')

    data_test = data_test.filter(
        regex='Age_.*|SibSp|Parch|Fare_.*|CabinLetter_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_id')

    data_train.to_csv(data_dir + "/processedtrain.csv", index=False)
    data_test.to_csv(data_dir + "/processedtest.csv", index=False)

    print("data process finished,output to processed*.csv")

    return

# return DataSets.TrainData
def read_train_data_sets(data_dir):
    if data_dir is None:
        return "the dir is none"

    data_train = pd.read_csv(data_dir + "/processedtrain.csv")
    train_np = data_train.as_matrix()
    train_Y = pd.get_dummies(data_train['Survived'], prefix='Survived').as_matrix()

    train_x = train_np[:600, 1:]
    train_y = train_Y[:600]
    train_set = DataSets.TrainData(train_x, train_y)

    return train_set

# return DataSets.ValidateData
def read_validation_data_sets(data_dir):
    if data_dir is None:
        return "the dir is none"

    data_train = pd.read_csv(data_dir + "/processedtrain.csv")
    train_np = data_train.as_matrix()
    train_Y = pd.get_dummies(data_train['Survived'], prefix='Survived').as_matrix()

    validate_x = train_np[600:, 1:]
    validate_y = train_Y[600:]
    validate_set = DataSets.ValidateData(validate_x, validate_y)

    return validate_set

# return DataSets.TestData
def read_test_data_sets(data_dir):
    if data_dir is None:
        return "the dir is none"

    data_test = pd.read_csv(data_dir + "/processedtest.csv")
    test_x = data_test.as_matrix()[:, 0:]
    test_set = DataSets.TestData(test_x, None)
    return test_set

# return DataSets
def read_all_data_sets(data_dir):
    if data_dir is None:
        return "the dir is none"

    data_train = pd.read_csv(data_dir + "/processedtrain.csv")
    data_test = pd.read_csv(data_dir + "/processedtest.csv")

    train_np = data_train.as_matrix()
    train_Y = pd.get_dummies(data_train['Survived'], prefix='Survived').as_matrix()

    train_x = train_np[:600, 1:]
    train_y = train_Y[:600]

    validate_x = train_np[600:, 1:]
    validate_y = train_Y[600:]

    test_x = data_test.as_matrix()[:, 0:]

    train_set = DataSets.TrainData(train_x, train_y)
    validate_set = DataSets.ValidateData(validate_x, validate_y)
    test_set = DataSets.TestData(test_x, None)

    data_sets = DataSets(train_set, validate_set, test_set)

    print("GET ALL DATASETS(TRAIN,VALIDATION,TEST)")

    return data_sets


# datasets = read_data_sets("./data")
# print(datasets.train_data.get_x())
# def main(argv=None):
#     data_url = "./data"
#     raw_data_process()


if __name__ == '__main__':
    data_url = "./data"
    raw_data_process(data_url)