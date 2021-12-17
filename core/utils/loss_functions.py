import numpy as np
# 使用多元对数损失函数计算分类的Loss
def multiclass_logloss(actual, predicted, eps=1e-15):
    """Logarithmic Loss  Metric
    :param actual: 包含actual target classes的数组, [[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0]], 每一行表示一个样本
    :param predicted: 分类预测结果矩阵, 每个类别都有一个概率 , 每一行表示一个样本
    """
    # 如果传入的actual是一维的数组，将其转换成和predicted数组形状一样的数组
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    # 裁剪概率范围，使得
    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    sigma = np.sum(actual * np.log(clip))
    return -1.0 / rows * sigma