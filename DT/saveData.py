from sklearn.datasets import load_iris
import pandas as pd

# 加载鸢尾花数据集
iris = load_iris()

# 将数据转换为 DataFrame
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# 将 DataFrame 保存为 CSV 文件
iris_df.to_csv('iris_data.csv', index=False)