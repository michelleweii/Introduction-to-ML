# Random Forest Algorithm on Iris Dataset
# -*- coding: UTF-8 -*-     
from random import seed
from random import randrange
from csv import reader
from math import sqrt

#随机森林在以决策树为基学习器构建Bagging集成的基础上，进一步在决策树的训练过程中引入了随机属性选择（即引入随机特征选择）。
#随机森林用于分类时，即采用n个决策树分类，将分类结果用简单投票法得到最终分类，提高分类准确率。
# 随机森林需要调整的参数有：
# （1）决策树的个数
# （2）特征属性的个数
# （3）递归次数（即决策树的深度）

# 加载数据
def load_csv(filename):
   dataset = list()
   with open(filename, 'r') as file:
       csv_reader = reader(file)
       for row in csv_reader:
           if not row:
               continue
           dataset.append(row)
   return dataset

# 除了判别列，其他列都转换为float类型
def str_column_to_float(dataset, column):
   for row in dataset:
       row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
   class_values = [row[column] for row in dataset]
   unique = set(class_values)
   lookup = dict()
   for i, value in enumerate(unique):
       lookup[value] = i
   for row in dataset:
       row[column] = lookup[row[column]]
   return lookup

# #将数据集分成N块，方便交叉验证
def cross_validation_split(dataset, n_folds):
   dataset_split = list()
   dataset_copy = list(dataset)
   fold_size = len(dataset) / n_folds
   for i in range(n_folds):
       fold = list()
       while len(fold) < fold_size:  #这里不能用if，if只是在第一次判断时起作用，while执行循环，直到条件不成立
           index = randrange(len(dataset_copy))
           fold.append(dataset_copy.pop(index))
           # pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
       dataset_split.append(fold)
   return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
   correct = 0
   for i in range(len(actual)):
       if actual[i] == predicted[i]:
           correct += 1
   return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
   folds = cross_validation_split(dataset, n_folds)
   scores = list()
   for fold in folds:
       train_set = list(folds)
       train_set.remove(fold)
       train_set = sum(train_set, [])
       test_set = list()
       for row in fold:
           row_copy = list(row)
           test_set.append(row_copy)
           row_copy[-1] = None
       predicted = algorithm(train_set, test_set, *args)
       actual = [row[-1] for row in fold]
       accuracy = accuracy_metric(actual, predicted)
       scores.append(accuracy)
   return scores

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
   left, right = list(), list()
   for row in dataset:
       if row[index] < value:
           left.append(row)
       else:
           right.append(row)
   return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, class_values):
   gini = 0.0
   for class_value in class_values:
       for group in groups:
           size = len(group)
           if size == 0:
               continue
           proportion = [row[-1] for row in group].count(class_value) / float(size)
           gini += (proportion * (1.0 - proportion))
   return gini

# 选取任意的n个特征，在这n个特征中，选取分割时的最优特征
# Select the best split point for a dataset
def get_split(dataset, n_features):
   class_values = list(set(row[-1] for row in dataset))
   b_index, b_value, b_score, b_groups = 999, 999, 999, None
   features = list()
   while len(features) < n_features:
       index = randrange(len(dataset[0])-1)
       if index not in features:
           features.append(index)
   for index in features:
       for row in dataset:
           groups = test_split(index, row[index], dataset)
           gini = gini_index(groups, class_values)
           if gini < b_score:
               b_index, b_value, b_score, b_groups = index, row[index], gini, groups
   return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
   outcomes = [row[-1] for row in group]
   return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
   left, right = node['groups']
   del(node['groups'])
   # check for a no split
   if not left or not right:
       node['left'] = node['right'] = to_terminal(left + right)
       return
   # check for max depth
   if depth >= max_depth:
       node['left'], node['right'] = to_terminal(left), to_terminal(right)
       return
   # process left child
   if len(left) <= min_size:
       node['left'] = to_terminal(left)
   else:
       node['left'] = get_split(left, n_features)
       split(node['left'], max_depth, min_size, n_features, depth+1)
   # process right child
   if len(right) <= min_size:
       node['right'] = to_terminal(right)
   else:
       node['right'] = get_split(right, n_features)
       split(node['right'], max_depth, min_size, n_features, depth+1)

# 构造决策树 Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
   root = get_split(dataset, n_features)
   split(root, max_depth, min_size, n_features, 1)
   return root

# Make a prediction with a decision tree
def predict(node, row):
   if row[node['index']] < node['value']:
       if isinstance(node['left'], dict):
           return predict(node['left'], row)
       else:
           return node['left']
   else:
       if isinstance(node['right'], dict):
           return predict(node['right'], row)
       else:
           return node['right']

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
   sample = list()
   n_sample = round(len(dataset) * ratio)
   while len(sample) < n_sample:
       index = randrange(len(dataset))
       sample.append(dataset[index])
   return sample

# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
   predictions = [predict(tree, row) for tree in trees]
   return max(set(predictions), key=predictions.count)

# 随机森林算法（多个决策树的组合）
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
   trees = list()
   for i in range(n_trees):
       sample = subsample(train, sample_size)
       tree = build_tree(sample, max_depth, min_size, n_features)
       trees.append(tree)
   predictions = [bagging_predict(trees, row) for row in test]
   return(predictions)

# Visualize model
# with open("allElectronicInformationGainOri.dot", 'w') as f:
#     f = tree.export_graphviz(
#         clf, feature_names=vec.get_feature_names(), out_file=f)

# Test the random forest algorithm
seed(1)
# load and prepare data
filename = 'irisdata.csv'
dataset = load_csv(filename)
#print dataset
#print '*****************************************************************'
# convert string attributes to integers
for i in range(0, len(dataset[0])-1):
   str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 5   # K 赋值为 5 用于交叉验证，得到每个子样本为 150/5 = 30，即超过 30 条返回记录会用于每次迭代时的评估。
max_depth = 10  #每棵树的最大深度设置为 10，
min_size = 1   #每个节点的最小训练行数为 1. 创建训练集样本的大小与原始数据集相同，这也是随机森林算法的默认预期值。
sample_size = 1.0
n_features = int(sqrt(len(dataset[0])-1))   #随机森林允许单个决策树使用特征的最大数量，//每个分裂点需要考虑的特征数
for n_trees in [1,5,10,15,30]:
   scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
   #print 出每组树的相应分值以及每种结构的平均分值
   print('Trees: %d' % n_trees)
   print('Scores: %s' % scores)
   print('Mean Accuracy: %.3f' %(sum(scores)/float(len(scores))))

# 代码实现流程：
# （1) 导入文件并将所有特征转换为float形式
# （2）将数据集分成n份，方便交叉验证
# （3）构造数据子集（随机采样），并在指定特征个数（假设m个，手动调参）下选取最优特征
# （4）构造决策树
# （5）创建随机森林（多个决策树的结合）
# （6）输入测试集并进行测试，输出预测结果

# 对以上代码的一点总结：
# 训练部分：假设我们取dataset中的m个feature来构造决策树，首先，我们遍历m个feature中的每一个feature，
# 再遍历每一行，通过spilt_loss函数（计算分割代价）来选取最优的特征及特征值，根据是否大于这个特征值进行
# 分类（分成left,right两类），循环执行上述步骤，直至不可分或者达到递归限值（用来防止过拟合），最后得到一个决策树tree。

# 测试部分：对测试集的每一行进行判断，决策树tree是一个多层字典，每一层为一个二分类，将每一行按
# 照决策树tree中的分类索引index一步一步向里层探索，直至不出现字典时探索结束，得到的值即为我们的预测值。