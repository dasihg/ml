from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集
# iris = load_iris()
# X = iris.data
# y = iris.target

# 加载 CSV 文件
data = pd.read_csv('iris_data.csv')
# 提取特征和目标变量
X = data.drop('target', axis=1)
y = data['target']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建决策树分类器
clf = tree.DecisionTreeClassifier()

# 在训练集上训练模型
clf.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = clf.predict(X_test)

# 计算模型准确率
accuracy = accuracy_score(y_test, y_pred)
print("模型准确率:", accuracy)

# 将测试集特征和预测结果合并为一个 DataFrame
test_result_df = pd.DataFrame(X_test)
test_result_df['Actual Target'] = y_test
test_result_df['Predicted Target'] = y_pred

# 将结果保存到新的 CSV 文件
test_result_df.to_csv('test_result.csv', index=True)

# 也可以在控制台打印测试集特征和预测结果
print(test_result_df)