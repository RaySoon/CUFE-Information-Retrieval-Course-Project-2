import csv
import re
import xlrd
import math
import numpy as np
import codecs
from scipy.stats import chi2_contingency
from gensim.models import Word2Vec

np.set_printoptions(suppress=True)

def sparseTxt(item_list, mode):   # 0,1,2
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

    url_idt=r"[W]{3}\.[\w]{2,}\.\w+\.*\w*"
    pat_affi=r"\.{2,}"

    count = 0
    # 辣鸡符号变空格，区分
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

            while re.search(pat_affi,temp):  # ..*n 归一化
                temp=temp.replace(re.search(pat_affi,temp).group()," ")
                pat_words.append("...")

            if len(temp)==0:
                pat_words.append('')
            for blocks in re.split("\s+", temp):   # 空格分块
                pointer = 0
                # if blocks==".":
                #     continue
                for matches in re.finditer(pattern,blocks):
                    span=matches.span()
                    if span[0]!=0 and blocks[pointer:span[0]] not in stop_list:
                        pat_words.append(blocks[pointer:span[0]])
                    # if re.findall(pat_affi,matches.group()):
                    #     pat_words.append("...")
                    # else:
                    #     pat_words.append(matches.group())
                    pointer=span[1]
                if pointer<len(blocks):  #剩下的文本
                    pat_words.append(blocks[pointer:])

            if count<5:
                print(pat_words)  # 测试效果
            count+=1
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
    # print(row, col)
    # print(row,len(result))
    result = sparseTxt(result, mode)
    return result, label


def initTrain(mode):  # neg,pos in list
    neg_train = open("../DATA/negative_train.txt", encoding='cp936').readlines()
    pos_train = open("../DATA/positive_train.txt", encoding='cp936').readlines()
    neg_len=len(neg_train)
    pos_len=len(pos_train)
    neg_train = sparseTxt(neg_train, mode)
    pos_train = sparseTxt(pos_train, mode)
    return neg_train, neg_len, pos_train, pos_len # 二维分词数组，title数


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
    # result_dict(word:word_count,word_doc_count),total word count
    return result, count


def calChi(pos_dict, postit_num, neg_dict, negdict_num, mode):  # 计算pos/neg 的 chisq 值
    arguments = np.zeros((2, 2))
    tit_sum = postit_num + negdict_num
    chi_dict = {}
    l_pos = list(pos_dict.keys())
    l_neg = list(neg_dict.keys())

    for word in list(set(l_neg + l_pos)):
        chi_dict[word] = 0.0
        if word in pos_dict.keys():
            arguments[1][1] = pos_dict[word][1]  # doc_count for word (N11)
        else:
            arguments[1][1] = 0
        arguments[0][1] = postit_num - arguments[1][1]

        if word in neg_dict.keys():
            arguments[1][0] = neg_dict[word][1]
        else:
            arguments[1][0] = 0
        arguments[0][0] = negdict_num - arguments[1][0]
        Nx_ = np.sum(arguments, axis=1)  # 各行加和
        N_x = np.sum(arguments, axis=0)  # 各列加和

        # print(arguments)

        # cache[word][0]=oneMI((0,1),arguments,Nx_,N_x)  #pos_MI

        chi_dict[word] = chi2_contingency(arguments)[0]  # pos_chisq
        arguments[:, [0, 1]] = arguments[:, [1, 0]]  # 交换列数
        # print(arguments)

        # Nx_=np.sum(arguments,axis=1)  # 各行加和
        # N_x=np.sum(arguments,axis=0)  # 各列加和
        # cache[word][1]=oneMI((0,1),arguments,Nx_,N_x) # neg MI

    # chisq-sorted
    sort_pos = sorted(chi_dict.items(), key=lambda d: d[1], reverse=True)
    data_write_csv("chisq_words_mode_" + str(mode) + ".csv", sort_pos)

    return sort_pos  # 返回chisq数组


def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow([data[0], data[1]])
    print("保存文件成功，处理结束")


def inChiTopK(word, chisq_tup):
    for items in chisq_tup:
        if word == items[0]:
            return True
    return False



def sumVecs(tits,model,dims):  #各个词的累加向量
    result=[]
    for title in tits:
        sum=np.zeros(dims)
        for words in title:
            if words in model.wv.vocab.keys():
                sum+=model.wv[words]
        result.append(sum)
    return result

def regularMatchEmail(email):
    pattern=re.match(r'\w+@([0-9a-zA-Z]+[-0-9a-zA-Z]*)(\.[0-9a-zA-Z]+[-0-9a-zA-Z]*)+',email,re.IGNORECASE)
    if pattern:
        print (email)
    else:
        print ("invalid eamil")

def toVec(mode, arg_dict, neg_train, pos_train, cut, test_tit):

    train_tit=neg_train+pos_train

    #注释决定哪些参数启用
    model=Word2Vec(size=arg_dict["size"],
                   # alpha=arg_dict["alpha"],
                   # min_count=arg_dict["min_count"],
                   # sg=arg_dict["alg"]
                   )
    model=Word2Vec(min_count=1)

    model.build_vocab(train_tit)
    model.train(train_tit, total_examples=model.corpus_count,
                epochs=model.iter,)
    print(model.corpus_count)
    vec_dim=arg_dict["size"]

    # inspection
    print(model.wv["SPRINTLINK"])
    # print(neg_model.wv["GENERAL"])

    # 向量累加
    print("向量累加:")
    train_suml=sumVecs(train_tit,model,vec_dim)
    return train_suml










if __name__ == '__main__':
    # toBayes(mode=1, p_pos=0.5, p_neg=0.5, word_size_range=(4400, 4800), step=50)

    # args for words2vec in dict:
    mode=1
    arg_dict={"size":100,        # 向量维度数
              "alpha":0,       # 学习率
              "min_count":2,   # 词频min_count以下不计入考虑范围
              "alg":1   # Training algorithm: 1 for skip-gram; otherwise CBOW.
              }
    # 初始化各路参数
    neg_train, negtit_num, pos_train, postit_num = initTrain(mode)  # l:标题数目
    test_tit, test_lab = initTest(mode + 1)
    #initial word vector
    vectors=toVec(mode=mode, arg_dict=arg_dict, cut=negtit_num, neg_train=neg_train,
          pos_train=pos_train, test_tit=test_tit)
    print(vectors[0])
    #SVM process