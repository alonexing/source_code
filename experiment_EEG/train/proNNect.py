import numpy as np
def proNNect(numDimension, rangeMatrx, width=0.1):
    """
    proNNect:用于生成指定的神经网络
    proNNect(numDimension, rangeMatrx, width = 0.1)
    输入:
        numDimension:神经网络维度
        rangeMatrx:生成的网络的范围，numDimension行，n列，第一列是上边界，第二列是下边界
        width:神经元间隔
    返回:
        Net:神经网络，字典对象
        Net['width']:神经网络神经元间隔
        Net['rangeMatrx']:神经网络各维度范围
        Net['numDimension']:神经网络维度
        Net['cent']:神经网络，行数代表神经元个数，列数代表神经网络维度
    """

    if numDimension < 1:
        print("错误：维度必须大于1")
        return

    rangeNum = np.zeros([numDimension, 2], dtype=np.int16)
    delta = 0.00
    rangeNum[:, 0] = np.ceil((rangeMatrx[:, 0] + delta) / width);
    rangeNum[:, 1] = np.floor((rangeMatrx[:, 1] - delta) / width);

    numPer = rangeNum[:, 0] - rangeNum[:, 1] + 1;
    numPer = np.append(numPer, 1)
    cent = np.zeros([np.prod(numPer), numDimension])

    for i in np.arange(0, numDimension, 1):
        for k in np.arange(0, np.prod(numPer[0:i + 1]), numPer[i]):
            for j in np.arange(rangeNum[i, 1], rangeNum[i, 0] + 1, 1):
                cent[(k + j - (rangeNum[i, 1])) * np.prod(numPer[i + 1: numDimension]): (k + j - (
                rangeNum[i, 1]) + 1) * np.prod(numPer[i + 1: numDimension]), i] \
                    = np.ones([np.prod(numPer[i + 1:numDimension])]) * j * width
    Net = {'width': width, 'rangeMatrx': rangeMatrx, 'numDimension': numDimension, 'cent': cent}
    return Net