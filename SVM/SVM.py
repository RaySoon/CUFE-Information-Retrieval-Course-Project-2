import re
import xlrd
import numpy as np
from sklearn import svm, metrics
from gensim.models import Word2Vec

np.set_printoptions(suppress=True)


def parseTitles(titleList, mode):
    ''''''

    # for a
    pattern = r'(,|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|=|\_|\+|，|。|、|；|‘|’|【|】|·|！|…|（|）|\*)+'
    stopList = ["VERY", "OURSELVES", "AM", "DOESN", "THROUGH", "ME", "AGAINST", "UP", "JUST", "HER", "OURS",
                "COULDN", "BECAUSE", "IS", "ISN", "IT", "ONLY", "IN", "SUCH", "TOO", "MUSTN", "UNDER", "THEIR",
                "IF", "TO", "MY", "HIMSELF", "AFTER", "WHY", "WHILE", "CAN", "EACH", "ITSELF", "HIS", "ALL",
                "ONCE", "HERSELF", "MORE", "OUR", "THEY", "HASN", "ON", "MA", "THEM", "ITS", "WHERE", "DID",
                "LL", "YOU", "DIDN", "NOR", "AS", "NOW", "BEFORE", "THOSE", "YOURS", "FROM", "WHO", "WAS",
                "M", "BEEN", "WILL", "INTO", "SAME", "HOW", "SOME", "OF", "OUT", "WITH", "S", "BEING", "T",
                "MIGHTN", "SHE", "AGAIN", "BE", "BY", "SHAN", "HAVE", "YOURSELVES", "NEEDN", "AND", "ARE", "O",
                "THESE", "FURTHER", "MOST", "YOURSELF", "HAVING", "AREN", "HERE", "HE", "WERE", "BUT", "THIS",
                "MYSELF", "OWN", "WE", "SO", "I", "DOES", "BOTH", "WHEN", "BETWEEN", "D", "HAD", "THE", "Y",
                "HAS", "DOWN", "OFF", "THAN", "HAVEN", "WHOM", "WOULDN", "SHOULD", "VE", "OVER", "THEMSELVES",
                "FEW", "THEN", "HADN", "WHAT", "UNTIL", "WON", "NO", "ABOUT", "ANY", "THAT", "SHOULDN",
                "DON", "DO", "THERE", "DOING", "AN", "OR", "AIN", "HERS", "WASN", "WEREN", "ABOVE",
                "AT", "YOUR", "THEIRS", "BELOW", "OTHER", "NOT", "RE", "HIM", "DURING", "WHICH"]

    urlFilter = r"[W]{3}\.[\w]{2,}\.\w+\.*\w*"
    dotFilter = r"\.{2,100}"

    if mode == 0:  # bi-gram
        result = [re.split("\s+", re.sub(pattern, " ", items.upper()))[:-1] for items in titleList]  # -1:\n存在于末尾

    else:  # 垃圾符号集计入分词列表中
        result = []
        for items in titleList:
            patWords = []


            if mode == 2:
                temp = items.upper().strip()  # csv文件不用去\n
            else:
                temp = items.upper()[:-1].strip()  # 去\n

            if len(temp) == 0:
                patWords.append('')
            else:
                temp=re.sub(urlFilter," .WEBSITE. ",temp)
                temp=re.sub(dotFilter," .DOT. ",temp)

                for blocks in re.split("\s+", temp):  # 空格分块
                    pointer = 0

                    for matches in re.finditer(pattern, blocks):
                        span = matches.span()
                        if span[0] != 0 and blocks[pointer:span[0]] not in stopList:
                        # if span[0] != 0 :
                            patWords.append(blocks[pointer:span[0]])
                        pointer = span[1]
                    if pointer < len(blocks):  # 剩下的文本
                        if blocks[pointer:] not in stopList:
                            patWords.append(blocks[pointer:])

            result.append(patWords)
    return result




def initTest(mode):
    result = []
    label = []
    workbook = xlrd.open_workbook('../DATA/testSet-1000.xlsx')
    booksheet = workbook.sheet_by_index(0)  # 用索引取第一个sheet
    row, col = booksheet.nrows, booksheet.ncols
    for rows in range(1, row):
        result.append(booksheet.cell_value(rows, 1))  # upload all test items
        label.append(booksheet.cell_value(rows, 2))
    result = parseTitles(result, mode)
    return result, label


def initTrain(mode):  # neg,pos in list
    negTrain = open("../DATA/negative_train.txt", encoding='cp936').readlines()
    posTrain = open("../DATA/positive_train.txt", encoding='cp936').readlines()
    negLen = len(negTrain)
    posLen = len(posTrain)
    negTrain = parseTitles(negTrain, mode)
    posTrain = parseTitles(posTrain, mode)
    return negTrain, negLen, posTrain, posLen  # 二维分词数组，title数


def meanVecs(tits, model, dims):  # 各个词的累加向量
    result = []
    count = 0
    for title in tits:
        sumCalculate = np.zeros(dims)
        for words in title:
            if words in model.wv.vocab.keys():
                sumCalculate += model.wv[words]
                count += 1
        tempList = (sumCalculate / count).tolist()
        result.append(tempList)
    result = np.array(result)
    return result


def toVec(argDict, negTrain, posTrain, testTitle):
    trainTitle = negTrain + posTrain
    cut = len(negTrain)

    # 注释决定哪些参数启用
    model = Word2Vec(size=argDict["size"],
                     # alpha=argDict["alpha"],
                     # sg=argDict["alg"],
                     min_count=argDict["min_count"]
                     )
    # model = Word2Vec(min_count=1)

    model.build_vocab(trainTitle)
    model.train(trainTitle, total_examples=model.corpus_count, epochs=model.iter)
    vectorDim = argDict["size"]

    # 向量平均
    print("向量平均:")
    trainMeanVector = meanVecs(trainTitle, model, vectorDim)
    testMeanVector = meanVecs(testTitle, model, vectorDim)
    trainLab = cut * ["N"] + cut * ["Y"]

    return trainMeanVector, trainLab, testMeanVector


if __name__ == '__main__':
    mode = 1
    argDict = {"size": 50,  # 向量维度数
                "alpha": 0,  # 学习率
                "min_count": 1,  # 词频min_count以下不计入考虑范围
                "alg": 1  # Training algorithm: 1 for skip-gram; otherwise CBOW.˚
               }
    # 初始化各路参数
    negTrain, negTitleNum, posTrain, posTitleNum = initTrain(mode)  # l:标题数目
    testTitle, testLabel = initTest(mode + 1)


    # initial word vector
    trainEntry, trainLabel, testEntry = toVec(argDict=argDict, negTrain=negTrain,
                                              posTrain=posTrain, testTitle=testTitle)
    print("vector convert finished")

    clf = svm.SVC(kernel='rbf', C=1,max_iter=1000)
    clf.fit(trainEntry, trainLabel)
    # pre = clf.predict(x_test)
    prediction = clf.predict(testEntry)
    print("SVM over")
    # 准确率
    score = metrics.accuracy_score(testLabel, prediction)
    print("准确率为：")
    print(score)