function [red_data] = KLD_reduce(data, num, s_eigvector)
    % 使用KLD得到的特征向量对数据进行降维与重建
    % 输入:
    %   data - 待处理数据，大小为 m x n（m个样本，n个特征）
    %   num - 选择的主成分数目
    %   s_eigvector - 主成分矩阵
    % 输出:
    %   red_data - 降维后的数据


    % 转换到新的特征空间
    reduced_data = data * s_eigvector(:, 1:num);  % 降维后的数据
    red_data = reduced_data;
 
end