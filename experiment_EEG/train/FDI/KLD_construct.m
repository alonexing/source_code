function [recon_data] = KLD_construct(data, num, s_eigvector)
    % 使用KLD得到的特征向量对数据进行降维与重建
    % 输入:
    %   data - 待处理数据，大小为 m x n（m个样本，n个特征）
    %   num - 选择的主成分数目
    %   s_eigvector - 主成分矩阵
    % 输出:
    %   recon_data - 重建后的数据

    reconstructed_data = data * s_eigvector(:, 1:num)'; % 还原后的数据
    recon_data = reconstructed_data; % 返回重建后的数据

end