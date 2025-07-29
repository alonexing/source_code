data_all = ssn.x;
avg_data = mean(data_all, 3);
% 假设 data 是你的数据矩阵
rows_to_exclude = [2, 18, 29 ,30];

% 创建所有行的索引
all_rows = 1:size(avg_data, 2);

% 排除指定的行
rows_to_keep = setdiff(all_rows, rows_to_exclude);

% 获取剩余的行
result = avg_data(:,rows_to_keep);
channels = 26;
Fs = 1000;
d = designfilt('bandpassiir', ...
               'FilterOrder', 4, ...
               'HalfPowerFrequency1',1, ...
               'HalfPowerFrequency2', 30, ...
               'SampleRate', Fs);
for ch = 1:channels
    result(:, ch) = filtfilt(d, result(:, ch));
end
taskdata = result(1:4000,:);
total_timepoints = 4000;
step = 500;
restdata = result(11001:15000,:);

% figure;
% topoplot(taskdata(:, 200), chanlocs, 'style', 'both', 'electrodes', 'labels', 'chaninfo', chaninfo);
% 
% figure;
% topoplot(restdata(:, 200), chanlocs, 'style', 'both', 'electrodes', 'labels', 'chaninfo', chaninfo);
num = 5;
[s_eigvector1, evr1] = KLDfeatures_cal(taskdata);
[s_eigvector2, evr2] = KLDfeatures_cal(restdata);
[task_red] = KLD_reduce(taskdata, num, s_eigvector1);
[rest_red] = KLD_reduce(restdata, num, s_eigvector2);

for t = 1:5
    m = task_red(:,t) * s_eigvector1(:, t)';
    n = rest_red(:,t) * s_eigvector2(:, t)';
    figure('Visible','off'); % 创建新窗口
    topoplot(s_eigvector1(:, t), chanlocs, 'style', 'both', 'electrodes', 'labels', 'chaninfo', chaninfo);
    title(sprintf('TaskData at component %d, explained variance: %.2f%%', t, evr1(t)*100));
    saveas(gcf, sprintf('Task_component%d.png', t));
    figure('Visible','off'); % 创建新窗口
    topoplot(s_eigvector2(:, t), chanlocs, 'style', 'both', 'electrodes', 'labels', 'chaninfo', chaninfo);
    title(sprintf('Rest Data at component %d, explained variance: %.2f%%', t, evr2(t)*100));
    saveas(gcf, sprintf('Rest_component%d.png', t));
end
for t = 1:5
    figure('Visible','off'); % 创建新窗口
    topoplot(s_eigvector1(:, t), chanlocs, 'style', 'both', 'electrodes', 'labels', 'chaninfo', chaninfo);
    h_labels = findobj(gca, 'Type', 'Text');
    set(h_labels, 'FontSize', 14);  % 例如设置为14号字体
    saveas(gcf, sprintf('Task_component%d.png', t));
    figure('Visible','off'); % 创建新窗口
    topoplot(s_eigvector2(:, t), chanlocs, 'style', 'both', 'electrodes', 'labels', 'chaninfo', chaninfo);
    h_labels = findobj(gca, 'Type', 'Text');
    set(h_labels, 'FontSize', 14);  % 例如设置为14号字体
    saveas(gcf, sprintf('Rest_component%d.png', t));
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

