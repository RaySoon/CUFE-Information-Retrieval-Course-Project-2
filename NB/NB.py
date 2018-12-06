"""
package 似乎清洗太多了反而正确率更低
from gensim.parsing.porter import PorterStemmer

nltk 清洗停用词也是同样的原因
from nltk.corpus import stopwords

"""

import csv
import re
import xlrd
import math
import numpy as np
import codecs
from scipy.stats import chi2_contingency

np.set_printoptions(suppress=True)


def parseTitles(titleList, mode):
    # p = PorterStemmer()
    # stopList = stopwords.words('english')

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
                "FEW", "THEN", "HADN", "WHAT", "UNTIL", "WON", "NO", "ABOUT", "ANY", "THAT", "FOR", "SHOULDN",
                "DON", "DO", "THERE", "DOING", "AN", "OR", "AIN", "HERS", "WASN", "WEREN", "ABOVE", "A",
                "AT", "YOUR", "THEIRS", "BELOW", "OTHER", "NOT", "RE", "HIM", "DURING", "WHICH"]

    urlFilter = r"[W]{3}\.[\w]{2,}\.\w+\.*\w*"
    dotFilter = r"\.{2,}"

    if mode == 0:
        result = [re.split("\s+", re.sub(pattern, " ", items.upper()))[:-1] for items in titleList]  # -1:\n存在于末尾

    else:  # 垃圾符号集计入分词列表中
        result = []
        for items in titleList:
            patWords = []

            if mode == 2:
                temp = items.upper().strip()  # csv文件不用去\n
            else:
                temp = items.upper()[:-1].strip()  # 去\n

            for matches in re.findall(urlFilter, temp):  # 网址归一化
                temp = temp.replace(matches, " ")
                patWords.append("#WEBSITE")

            while re.search(dotFilter, temp):  # ..*n 归一化
                temp = temp.replace(re.search(dotFilter, temp).group(), " ")
                patWords.append("...")

            if len(temp) == 0:
                patWords.append('')
            for blocks in re.split("\s+", temp):  # 空格分块
                pointer = 0
                for matches in re.finditer(pattern, blocks):
                    span = matches.span()
                    if span[0] != 0 and blocks[pointer:span[0]] not in stopList:
                        patWords.append(blocks[pointer:span[0]])
                    pointer = span[1]
                if pointer < len(blocks):  # 剩下的文本
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


def createDict(listItems):
    result = {}
    count = 0
    for items in listItems:
        temp = []
        for words in items:
            if words not in result.keys():
                result[words] = [1, 1]  # word: word count; doc count
            else:
                result[words][0] += 1  # 出现的词汇数+1
                if words not in temp:  # 此title已经加过文档数
                    result[words][1] += 1  # 出现的文档数+1
            temp.append(words)
            count += 1
    return result, count


def calChi(posDict, posTitleNumber, negDict, negDictNum, mode):  # 计算pos/neg 的 chisq 值
    arguments = np.zeros((2, 2))
    chisqDict = {}
    posWords = list(posDict.keys())
    negWords = list(negDict.keys())

    for word in list(set(negWords + posWords)):
        chisqDict[word] = 0.0

        arguments[1][1] = posDict[word][1] if word in posDict.keys() else 0
        arguments[0][1] = posTitleNumber - arguments[1][1]
        arguments[1][0] = negDict[word][1] if word in negDict.keys() else 0
        arguments[0][0] = negDictNum - arguments[1][0]

        chisqDict[word] = chi2_contingency(arguments)[0]  # pos_chisq

    sortPos = sorted(chisqDict.items(), key=lambda d: d[1], reverse=True)
    dataWriteCsv("chisq_words_mode_" + str(mode) + ".csv", sortPos)

    return sortPos  # 返回chisq数组


def dataWriteCsv(fileName, datas):
    csvFile = codecs.open(fileName, 'w+', 'utf-8')  # 追加
    writer = csv.writer(csvFile, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow([data[0], data[1]])
    print("保存文件成功")


def inChiTopK(word, chisqTuple):
    for items in chisqTuple:
        if word == items[0]:
            return True
    return False


def toBayes(posProb, negProb, wordSize, chisqIndex, posWords, negWords, testLabel):
    resultList = []
    chisqRange = chisqIndex[:wordSize]
    pinLab = 0
    countTrue = 0
    countFalse = 0
    for titles in testTitle:
        negNB = math.log2(posProb)
        posNB = math.log2(negProb)
        for words in titles:
            if inChiTopK(words, chisqRange):

                if words in posWords.keys():
                    posNB += math.log2((posWords[words][0] + 1) / (posWordCount + posTypes))  # 加一平滑
                else:
                    posNB += math.log2(1 / (posWordCount + posTypes))
                if words in negWords.keys():
                    negNB += math.log2((negWords[words][0] + 1) / (negWordCount + negTypes))
                else:
                    negNB += math.log2(1 / (negWordCount + negTypes))

        judge = "Y" if posNB > negNB else "N"

        if judge == testLabel[pinLab]:
            countTrue += 1
        else:
            countFalse += 1
            resultList.append([str(titles) + ",", str(judge) + ",", str(testLabel[pinLab])])
        pinLab += 1
    print("error ", countFalse)


if __name__ == '__main__':
    mode = 1
    negTrain, negTitleNum, posTrain, posTitleNum = initTrain(mode)  # l:标题数目
    testTitle, testLab = initTest(mode + 1)
    negWords, negWordCount = createDict(negTrain)  # count:number of words
    posWords, posWordCount = createDict(posTrain)  # create dicts of neg & pos
    negTypes = len(negWords.keys())
    posTypes = len(posWords.keys())

    chisqIndex = calChi(posWords, posTitleNum, negWords, negTitleNum, mode)  # tup: (word,[chisq])

    for i in range(4400, 4800, 100):
        print(i)
        toBayes(0.5, 0.5, i, chisqIndex, posWords, negWords, testLab)
