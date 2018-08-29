# -*- coding: UTF-8 -*-   
from numpy import *

#创建一个包含在所有文档中出现的词的列表（词不重复）
def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)

def spamTest():
    docList = [];
    classList = [];
    fullText = []
    for i in range(1, 26):
        # 读取垃圾邮件
        wordList = textParse(open('email/spam/%d.txt' % i).read())  # 把每封邮件中的内容转成字符串，然后拆分成单词装入List
        docList.append(wordList)
        classList.append(1)  # 垃圾邮件的标签是1
        # 读取正常邮件
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(0)  # 正常邮件的标签是0
    vocabList = createVocabList(docList)  # 创建词汇表
    trainingSet = range(50);  # 训练集
    testSet = []  # 测试集
    #  随机从训练集中的50条数据中选取10条作为测试集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = [];
    trainClasses = []  # 训练集矩阵；训练集标签
    for docIndex in trainingSet:  #循环遍历训练集的所有文档，基于词汇表，构建词向量
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))  # 将训练集中的每一条数据，转化为词向量
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))  # 开始训练
    # 用10条测试数据，测试分类器的准确性
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])  # 打印输出错判的那条数据
    print('the error rate is: ', float(errorCount) / len(testSet))  # 错误率  #将垃圾邮件误判为正常邮件
    # return vocabList,fullText