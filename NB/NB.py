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


def sparseTxt(item_list, mode):
    # p = PorterStemmer()
    # stop_list = stopwords.words('english')

    pattern = r'(,|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|=|\_|\+|，|。|、|；|‘|’|【|】|·|！|…|（|）|\*)+'
    stop_list = ["VERY", "OURSELVES", "AM", "DOESN", "THROUGH", "ME", "AGAINST", "UP", "JUST", "HER", "OURS",
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

    url_idt = r"[W]{3}\.[\w]{2,}\.\w+\.*\w*"
    pat_affi = r"\.{2,}"

    if mode == 0:
        result = [re.split("\s+", re.sub(pattern, " ", items.upper()))[:-1] for items in item_list]  # -1:\n存在于末尾

    else:  # 垃圾符号集计入分词列表中
        result = []
        for items in item_list:
            if mode == 2:
                temp = items.upper().strip()  # csv文件不用去\n
            else:
                temp = items.upper()[:-1].strip()  # 去\n
            pat_words = []

            for matches in re.findall(url_idt, temp):  # 网址归一化
                temp = temp.replace(matches, " ")
                pat_words.append("~WEBSITE~")

            while re.search(pat_affi, temp):  # ..*n 归一化
                temp = temp.replace(re.search(pat_affi, temp).group(), " ")
                pat_words.append("...")

            if len(temp) == 0:
                pat_words.append('')
            for blocks in re.split("\s+", temp):  # 空格分块
                pointer = 0
                for matches in re.finditer(pattern, blocks):
                    span = matches.span()
                    if span[0] != 0 and blocks[pointer:span[0]] not in stop_list:
                        pat_words.append(blocks[pointer:span[0]])
                    pointer = span[1]
                if pointer < len(blocks):  # 剩下的文本
                    pat_words.append(blocks[pointer:])

            result.append(pat_words)
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
    result = sparseTxt(result, mode)
    return result, label


def initTrain(mode):  # neg,pos in list
    neg_train = open("../DATA/negative_train.txt", encoding='cp936').readlines()
    pos_train = open("../DATA/positive_train.txt", encoding='cp936').readlines()
    neg_len = len(neg_train)
    pos_len = len(pos_train)
    neg_train = sparseTxt(neg_train, mode)
    pos_train = sparseTxt(pos_train, mode)
    return neg_train, neg_len, pos_train, pos_len  # 二维分词数组，title数


def createDict(list_items):
    result = {}
    count = 0
    for items in list_items:
        temp = []
        for words in items:
            if words not in result.keys():
                result[words] = [1, 1]  # word: word count; doc_count
            else:
                result[words][0] += 1  # 出现的词汇数+1
                if words not in temp:  # 此title已经加过文档数
                    result[words][1] += 1  # 出现的文档数+1
            temp.append(words)
            count += 1
    return result, count


def calChi(pos_dict, postit_num, neg_dict, negdict_num, mode):  # 计算pos/neg 的 chisq 值
    arguments = np.zeros((2, 2))
    chi_dict = {}
    l_pos = list(pos_dict.keys())
    l_neg = list(neg_dict.keys())

    for word in list(set(l_neg + l_pos)):
        chi_dict[word] = 0.0

        arguments[1][1] = pos_dict[word][1] if word in pos_dict.keys() else 0
        arguments[0][1] = postit_num - arguments[1][1]
        arguments[1][0] = neg_dict[word][1] if word in neg_dict.keys() else 0
        arguments[0][0] = negdict_num - arguments[1][0]

        chi_dict[word] = chi2_contingency(arguments)[0]  # pos_chisq

    sort_pos = sorted(chi_dict.items(), key=lambda d: d[1], reverse=True)
    data_write_csv("chisq_words_mode_" + str(mode) + ".csv", sort_pos)

    return sort_pos  # 返回chisq数组


def data_write_csv(file_name, datas):
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow([data[0], data[1]])
    print("保存文件成功")


def inChiTopK(word, chisq_tup):
    for items in chisq_tup:
        if word == items[0]:
            return True
    return False


def toBayes(p_pos, p_neg, word_size, chi_ori_tup, pos_words, neg_words):
    result_list = []
    decision = chi_ori_tup[:word_size]
    pin_lab = 0
    count_true = 0
    count_false = 0
    for titles in test_tit:
        nb_neg = math.log2(p_pos)
        nb_pos = math.log2(p_neg)
        for words in titles:
            if inChiTopK(words, decision):

                if words in pos_words.keys():
                    nb_pos += math.log2((pos_words[words][0] + 1) / (poswordcount + pos_types))  # 加一平滑
                else:
                    nb_pos += math.log2(1 / (poswordcount + pos_types))
                if words in neg_words.keys():
                    nb_neg += math.log2((neg_words[words][0] + 1) / (negwordcount + neg_types))
                else:
                    nb_neg += math.log2(1 / (negwordcount + neg_types))

        judge = "Y" if nb_pos > nb_neg else "N"

        if judge == test_lab[pin_lab]:
            count_true += 1
        else:
            count_false += 1
            result_list.append([str(titles) + ",", str(judge) + ",", str(test_lab[pin_lab])])
        pin_lab += 1
    print("error ", count_false)


if __name__ == '__main__':
    mode = 1
    neg_train, negtit_num, pos_train, postit_num = initTrain(mode)  # l:标题数目
    test_tit, test_lab = initTest(mode + 1)
    neg_words, negwordcount = createDict(neg_train)  # count:number of words
    pos_words, poswordcount = createDict(pos_train)  # create dicts of neg & pos
    neg_types = len(neg_words.keys())
    pos_types = len(pos_words.keys())

    chi_ori_tup = calChi(pos_words, postit_num, neg_words, negtit_num, mode)  # tup: (word,[chisq])

    for i in range(4400, 4800, 100):
        print(i)
        toBayes(0.5, 0.5, i, chi_ori_tup, pos_words, neg_words)
