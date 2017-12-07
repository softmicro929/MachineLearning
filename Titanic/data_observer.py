import pandas as pd
import matplotlib.pyplot as plt

def data_observe(data_dir):
    df = pd.read_csv(data_dir+"/train.csv")
    print("the train_data column:%d"%(df.shape[0]))
    # columns = df.columns
    # for i in columns:
    #     print("%15s"%(i),"  missing %5d"%(pd.isnull(df[i]).sum()), " type: ", df[i].dtypes)
    #
    # #print(df.head())
    #
    # # # 平均值：
    # # df['Fare'][ np.isnan(df['Fare']) ] = df['Fare'].median()
    # # 最常见的值：
    # # # dadf.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().valuesta_tain,data_test = read_data_sets("./data")
    # # Replace missing values with "U0"
    # #
    # #大概意思就是首先将空的填补成U0，然后用正则匹配所有的Cabin项，将Cabin的字母+数字组合转为只有字母，过滤掉数字
    # # 然后用factorize将字母转换为不同的数字，将类别较多的标称属性转化为数字，这个对应于Dummy Variables，（类别少）
    # df['Cabin'][df.Cabin.isnull()] = 'U0'
    # # create feature for the alphabetical part of the cabin number
    # df['CabinLetter'] = df['Cabin'].map( lambda x : re.compile("([a-zA-Z]+)").search(x).group())
    #
    # # factorize函数可以将Series中的标称型数据映射称为一组数字，相同的标称型映射为相同的数字。
    # # factorize函数的返回值是一个tuple（元组），元组中包含两个元素。
    # # 第一个元素是一个array，其中的元素是标称型元素映射为的数字；
    # # 第二个元素是Index类型，其中的元素是所有标称型元素，没有重复。
    # df['CabinLetter'] = pd.factorize(df['CabinLetter'])[0]
    # #print(df)
    #
    # df['Fare_bin'] = pd.qcut(df['Fare'], 4)
    #
    # # qcut() creates a new variable that identifies the quartile range, but we can't use the string so either
    # # factorize or create dummies from the result
    # # df['Fare_bin_id'] = pd.factorize(df['Fare_bin'])[0]
    # df = pd.concat([df, pd.get_dummies(df['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))], axis=1)
    #
    # print(df)
    #
    # fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,5))
    # axis1.set_title("Origin age values")
    # axis2.set_title("New age values")
    #
    # average_age = df['Age'].mean()
    # df['Age'].plot(kind='hist',bins=70,ax=axis1)
    # df['Age'][df.Age.isnull()] = average_age
    # df['Age'].plot(kind='hist',bins=70,ax=axis2)
    # plt.show()

def pdata_observe(data_dir):
    df = pd.read_csv(data_dir + "/processedtrain.csv")
    fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,5))
    axis1.set_title("Origin age values")
    axis2.set_title("New age values")

    average_age = df['Age'].mean()
    df['Age'].plot(kind='hist',bins=70,ax=axis1)
    df['Age'][df.Age.isnull()] = average_age
    df['Age'].plot(kind='hist',bins=70,ax=axis2)
    plt.show()

# data_observe("./data")
# data_train = pd.read_csv("./data/train.csv")
# trainset_process(data_train)
# print('finished')