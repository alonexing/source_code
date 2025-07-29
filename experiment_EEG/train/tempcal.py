from collections.abc import Iterable
import numpy as np
def tempcal(inputData, cent, TS, eta, kk, alpha, gamma, sigma, keta, boolWNorm, splicejoint):
    [M, colCent] = np.shape(cent)
    [Steps, colInput] = np.shape(inputData)
    # a = 0.05
    pj = 500
    inputData = inputData.astype(np.float32)
    pa = np.ones(shape=(1, M), dtype=np.int32)
    # x_repeat = np.tile(inputData, (kk, 1))
    x_repeat = np.array([inputData for i in range(kk)], dtype=np.float32).reshape(-1, colCent)


    # print(splicejoint)        if i >= ST_steps - pj:
    #             V[i + pj - ST_steps, :, :] = W
    if not splicejoint is None:
        splicejoint = [np.array(splicejoint) + i * Steps for i in range(kk)]  # 拼接点也要重复
        splicejoint = np.concatenate(splicejoint)
    # print(splicejoint)

    ST_steps = x_repeat.shape[0]
    # ST_steps = np.size(x_repeat, axis = 0)
    x_hat = np.zeros([ST_steps, colCent])
    S1 = np.array(np.zeros([M, 1]))
    W = np.zeros([M, colCent])
    V = np.zeros([pj, M, colCent])
    errors = np.zeros((ST_steps - 1, colCent))
    if boolWNorm:
        WNorm = np.zeros((M, ST_steps - 1))
    else:
        WNorm = []

    SS = np.zeros((M, Steps));

    for j in np.arange(0, Steps, 1):
        # S1[j] = np.exp(-np.sum((x_repeat[0, :] - cent[j, :])**2)/(keta * (eta**2)))
        temp = np.transpose(pa) * x_repeat[j, :] - cent
        SS[:, j] = np.exp(-np.sum(temp ** 2, axis=1) / (keta * (eta ** 2)))
        # tempdata = 1 + np.dot(np.transpose(S1), S1) * sigma

    for i in np.arange(1, ST_steps - 1, 1):
        # temp = np.transpose(pa) * x_repeat[i-1, : ] - cent
        # S2 = np.exp(-np.sum(temp**2, axis = 1)/(2 * (eta**2)))
        # tempdata = 1 + np.dot(np.transpose(S1), S1) * sigma
        ii = (i + 1) % Steps

        ii = Steps - 1 if ii == 0 else ii

        iim = ii - 1
        iim = Steps - 1 if iim == 0 else iim
        S2 = SS[:, iim]
        #
        # x_hat[i, :] = x_hat[i - 1, :] + (alpha - 1) * (x_hat[i - 1, :] - x_repeat[i - 1, :]) + TS * np.dot(S2, W)
        # x_hat[i, :] =  0.1* (x_hat[i - 1, :] - x_repeat[i - 1, :]) + np.dot(S2, W)
        x_hat[i, :] = alpha * (x_hat[i - 1, :] - x_repeat[i - 1, :]) + np.dot(S2, W)
        # 判断是否为拼接点，如果为拼接点，则将真实值赋予预测值
        if i % Steps == 0 or (isinstance(splicejoint, Iterable) and (i in splicejoint)):
            # i=i+2
            x_hat[i, :] = x_repeat[i, :]
            #
            # errors[i - 1, :] = (np.reshape(x_hat[i, :] - x_repeat[i, :], (1, -1)))
            # continue
        errors[i - 1, :] = (np.reshape(x_hat[i, :] - x_repeat[i, :], (1, -1)))
        # errors[i, :] = (np.reshape(x_hat[i, :] - x_repeat[i, :], (1, -1)))
        W = W - gamma * np.dot(np.reshape(S2, (-1, 1)),
                                    np.reshape(x_hat[i, :] - x_repeat[i, :], (1, -1)))
        # W = W - gamma * np.dot(np.reshape(S2, (-1, 1)),
        #                             np.reshape(x_hat[i, :] - x_repeat[i, :], (1, -1))) - sigma * W
        # print(W.shape)
        if boolWNorm:
            WNorm[:, i] = np.linalg.norm(W, ord=2, axis=1)

        if i >= ST_steps - pj:
            V[i + pj - ST_steps, :, :] = W

    S = np.zeros([M, Steps])
    for j in np.arange(0, Steps, 1):
        temp = np.transpose(pa) * inputData[j, :] - cent
        S[:, j] = np.exp(-np.sum(np.square(temp), axis=1) / (1 * np.square(eta)))

    W_BAR = np.transpose(np.mean(V, axis=0))
    WS = np.dot(W_BAR, S)
    W_BAR1=W_BAR.T
    for i in np.arange(1, ST_steps - 1, 1):
        # temp = np.transpose(pa) * x_repeat[i-1, : ] - cent
        # S2 = np.exp(-np.sum(temp**2, axis = 1)/(2 * (eta**2)))
        # tempdata = 1 + np.dot(np.transpose(S1), S1) * sigma
        ii = (i + 1) % Steps
        ii = Steps - 1 if ii == 0 else ii
        iim = ii - 1
        iim = Steps - 1 if iim == 0 else iim
        S2 = S[:, iim]

        # x_hat[i, :] = x_hat[i - 1, :] + (alpha - 1) * (x_hat[i - 1, :] - x_repeat[i - 1, :]) + TS * np.dot(S2, W)
        x_hat[i, :] = alpha * (x_hat[i - 1, :] - x_repeat[i - 1, :]) + np.dot(S2, W)
        # x_hat[i, :] = 0.1 * (x_hat[i - 1, :] - x_repeat[i - 1, :]) + TS * np.dot(S2, W_BAR1)
        # x_hat[i, :] = 0.1 * (x_hat[i - 1, :] - x_repeat[i - 1, :]) + np.dot(S2, W_BAR1)
        errors1 = x_hat-x_repeat
    return (W_BAR, WS, S, WNorm, errors,x_hat,errors1)


