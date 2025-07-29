function [s_eigvector, evr] = KLDfeatures_cal(data)
    % 计算PCA特征向量和解释方差比
    % 输入:
    %   data - 原始数据，大小为 m x n（m个样本，n个特征）
    % 输出:
    %   s_eigvector - 排序后的特征向量矩阵
    %   evr - 每个主成分的解释方差比


    % 计算协方差矩阵
    cov_matrix = cov(data);  % 计算协方差矩阵

    % 特征值分解
    [eigenvectors, eigenvalues] = eig(cov_matrix);  % 计算特征值和特征向量

    % 特征值排序
    eigenvalues = diag(eigenvalues);  % 将特征值转换为列向量
    [~, index] = sort(eigenvalues, 'descend');  % 排序索引
    sorted_eigenvectors = eigenvectors(:, index);  % 排序后的特征向量
    sorted_eigenvalues = eigenvalues(index);  % 排序后的特征值
    s_eigvector = sorted_eigenvectors;
    % 计算解释方差比
    total_variance = sum(sorted_eigenvalues);  % 总方差
    explained_variance_ratio = sorted_eigenvalues / total_variance;  % 解释方差比
    evr = explained_variance_ratio;
end