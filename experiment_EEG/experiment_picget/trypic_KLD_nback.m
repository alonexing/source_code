data_back0 = ssn_0back.x;
avg_data_back0 = mean(data_back0, 3);
data_back2 = ssn_2back.x;
avg_data_back2 = mean(data_back2, 3);
% 假设 data 是你的数据矩阵
rows_to_exclude = [2, 18, 29 ,30];

% 创建所有行的索引
all_rows = 1:size(avg_data_back0, 2);

% 排除指定的行
rows_to_keep = setdiff(all_rows, rows_to_exclude);
% 获取剩余的行
result_back0 = avg_data_back0(:,rows_to_keep);
result_back2 = avg_data_back2(:,rows_to_keep);
back0data = result_back0(1:4000,:);
total_timepoints = 4000;
step = 500;
back2data = result_back2(1:4000,:);

% figure;
% topoplot(taskdata(:, 200), chanlocs, 'style', 'both', 'electrodes', 'labels', 'chaninfo', chaninfo);
% 
% figure;
% topoplot(restdata(:, 200), chanlocs, 'style', 'both', 'electrodes', 'labels', 'chaninfo', chaninfo);
[s_eigvector1, evr1] = KLDfeatures_cal(back0data);
[s_eigvector2, evr2] = KLDfeatures_cal(back2data);
for t = 1:5
    figure('Visible','off'); % 创建新窗口
    topoplot(s_eigvector1(:, t), chanlocs, 'style', 'both', 'electrodes', 'labels', 'chaninfo', chaninfo);
    h_labels = findobj(gca, 'Type', 'Text');
    set(h_labels, 'FontSize', 14);  % 例如设置为14号字体
    saveas(gcf, sprintf('BACK0_component%d.png', t));
    figure('Visible','off'); % 创建新窗口
    topoplot(s_eigvector2(:, t), chanlocs, 'style', 'both', 'electrodes', 'labels', 'chaninfo', chaninfo);
    h_labels = findobj(gca, 'Type', 'Text');
    set(h_labels, 'FontSize', 14);  % 例如设置为14号字体
    saveas(gcf, sprintf('BACK2_component%d.png', t));
end

% 循环每隔500个时间点绘图
% for t = 1:step:total_timepoints
%     figure('Visible','off'); % 创建新窗口
%     topoplot(taskdata(:, t), chanlocs, 'style', 'both', 'electrodes', 'labels', 'chaninfo', chaninfo);
%     title(sprintf('Task Data at Time Point %d', t));
%     saveas(gcf, sprintf('Task_Time%d.png', t));
%     figure('Visible','off'); % 创建新窗口
%     topoplot(restdata(:, t), chanlocs, 'style', 'both', 'electrodes', 'labels', 'chaninfo', chaninfo);
%     title(sprintf('Rest Data at Time Point %d', t));
%     saveas(gcf, sprintf('Rest_Time%d.png', t));
% end

