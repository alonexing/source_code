import numpy as np
from tempcal import tempcal
def DeterLearningnew(inputData, cent, propertypData, boolWNorm=True, splicejoint=None):
    """
    确定学习理论种的模式训练阶段
    使用伟明新算法
    DeterLearning(inputData, cent, propertypData)
    输入:
        inputData:待训练的模式,每一行是一个数据点,行数代表数据点个数，列数代表维度
        cent:神经网络,维度与inputdata保持一致,每一行是一个神经网络点，行数代表神经元个数，列数代表神经网络维度
        propertypData:学习参数配置
        propertypData['TS']:周期
        propertyData['eta']:RBF函数的激活半径
        propertyData['repeat']:数据迭代周期数
        propertyData['alpha']:数据学习速率

    输出:
        输出元组
        W_BAR:输出权值矩阵,列数代表神经网络个数，行数代表维度
        WS:动态模型，列数代表数据点数，行数代表维度
    """

    if isinstance(propertypData, (dict)):
        try:
            TS = propertypData['TS']
            eta = propertypData['eta']
            kk = propertypData['repeat']
            alpha = propertypData['alpha']
            gamma = propertypData['gamma']
            sigma = propertypData['sigma']
            keta = propertypData['keta']

        except KeyError:
            print("错误：请设置参数：TS, eta, repeat, alpha")
            return 1
    else:
        print("错误：propertypData应该是一个字典类型，请检测数据类型！")

    [M, colCent] = np.shape(cent)
    [Steps, colInput] = np.shape(inputData)

    if colCent != colInput:
        print("错误：请确认神经网络维度与输入变量维度一致，cent与inputData的列必须保持一致！")
        return

    return tempcal(inputData, cent, TS, eta, kk, alpha, gamma, sigma, keta, boolWNorm=boolWNorm,
                   splicejoint=splicejoint)