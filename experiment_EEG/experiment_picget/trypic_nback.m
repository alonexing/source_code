data_back0 = ssn_0back.x;
avg_data_back0 = mean(data_back0, 3);
data_back2 = ssn_2back.x;
avg_data_back2 = mean(data_back2, 3);
% 假设 data 是你的数据矩阵
rows_to_exclude = [2, 18, 29 ,30];
chaninfo.labelfontSize = 24;  % 增大字体
% 创建所有行的索引
all_rows = 1:size(avg_data_back0, 2);

% 排除指定的行
rows_to_keep = setdiff(all_rows, rows_to_exclude);
% 获取剩余的行
result_back0 = avg_data_back0(:,rows_to_keep);
result_back2 = avg_data_back2(:,rows_to_keep);
back0data = result_back0(1:4000,:)';
total_timepoints = 4000;
step = 200;
back2data = result_back2(1:4000,:)';
% 循环每隔500个时间点绘图
% for t = 1:step:total_timepoints
%     figure('Visible','off'); % 创建新窗口
%     topoplot(back0data(:, t), chanlocs, 'style', 'both', 'electrodes', 'labels', 'chaninfo', chaninfo);
%     title(sprintf('back0 Data at Time Point %d', t));
%     saveas(gcf, sprintf('back0_Time%d.png', t));
%     figure('Visible','off'); % 创建新窗口
%     topoplot(back2data(:, t), chanlocs, 'style', 'both', 'electrodes', 'labels', 'chaninfo', chaninfo);
%     title(sprintf('back1 Data at Time Point %d', t));
%     saveas(gcf, sprintf('back1_Time%d.png', t));
% end
% 定义频段（示例）
freq_bands = {
    'Delta', [1, 4];
    'Theta', [4, 8];
    'Alpha', [8, 13];
    'Beta', [13, 30];
};
fs = 250; % 采样率
window_length_sec = 0.8; % 以秒为单位
window_samples = round(window_length_sec * fs); % 每个窗口的采样点数
noverlap = window_samples / 2; % 50%重叠
nfft = 2^nextpow2(window_samples); % FFT点数

% 计算 taskdata 每个窗口的频段功率
num_windows_back0 = floor(size(back0data,2) / window_samples);
num_windows_back2 = floor(size(back2data,2) / window_samples);
n_channels = size(back0data, 1);

% 初始化存储
task_power = struct();
rest_power = struct();

for k = 1:size(freq_bands, 1)
    band_name = freq_bands{k, 1};
    task_power.(band_name) = zeros(n_channels, num_windows_back0);
    rest_power.(band_name) = zeros(n_channels, num_windows_back2);
end

% 计算 taskdata 每个窗口的频段功率
for w = 1:num_windows_back0
    idx_start = (w-1)*window_samples + 1;
    idx_end = w*window_samples;
    data_segment_task = back0data(:, idx_start:idx_end); % 维度：通道 x 时间
    for ch = 1:n_channels
        channelData = data_segment_task(ch, :)';
        [S, F, ~] = spectrogram(channelData, window_samples, noverlap, nfft, fs);
        P = abs(S).^2; % 频谱的功率
        for k = 1:size(freq_bands, 1)
            band_name = freq_bands{k, 1};
            f_range = freq_bands{k, 2};
            freq_mask = (F >= f_range(1)) & (F <= f_range(2));
            mean_power = mean(P(freq_mask, :), 1); % 频段内所有频点的平均
            back0_power.(band_name)(ch, w) = mean(mean_power);
        end
    end
end

% 计算 restdata 每个窗口的频段功率
for w = 1:num_windows_back2
    idx_start = (w-1)*window_samples + 1;
    idx_end = w*window_samples;
    data_segment_rest = back2data(:, idx_start:idx_end);
    for ch = 1:n_channels
        channelData = data_segment_rest(ch, :)';
        [S, F, ~] = spectrogram(channelData, window_samples, noverlap, nfft, fs);
        P = abs(S).^2;
        for k = 1:size(freq_bands, 1)
            band_name = freq_bands{k, 1};
            f_range = freq_bands{k, 2};
            freq_mask = (F >= f_range(1)) & (F <= f_range(2));
            mean_power = mean(P(freq_mask, :), 1);
            back2_power.(band_name)(ch, w) = mean(mean_power);
        end
    end
end
% 定义参数
num_windows_back0 = size(back0_power.(band_name), 2);
num_windows_back2 = size(back2_power.(band_name), 2);

% 画图（以任务数据为例）
for w = 1:num_windows_back0
    for k = 1:size(freq_bands, 1)
        band_name = freq_bands{k, 1};
        % 提取第w个窗口的频段平均功率
        data_back0_band = back0_power.(band_name)(:, w);  % 所有通道
        data_back2_band = back2_power.(band_name)(:, w);

        % 绘制Task脑地形图
        figure('Visible','off');
        topoplot(data_back0_band, chanlocs, 'style', 'both', 'electrodes', 'labels', 'chaninfo', chaninfo);
        h_labels = findobj(gca, 'Type', 'Text');
        set(h_labels, 'FontSize', 14);  % 例如设置为14号字体
        % title(sprintf('back0 %s Band at Window %d', band_name, w));
        saveas(gcf, sprintf('back0_%s_Window%d.png', band_name, w));
        close;

        % 绘制Rest脑地形图
        figure('Visible','off');
        topoplot(data_back2_band, chanlocs, 'style', 'both', 'electrodes', 'labels', 'chaninfo', chaninfo);
        % title(sprintf('back2 %s Band at Window %d', band_name, w));
        h_labels = findobj(gca, 'Type', 'Text');
        set(h_labels, 'FontSize', 14);  % 例如设置为14号字体
        saveas(gcf, sprintf('back2_%s_Window%d.png', band_name, w));
        close;
    end
end
% 先绘制第一个头皮图
% figure;
% topoplot(taskdata(:, 200), chanlocs, 'style', 'both', 'electrodes', 'labels', 'chaninfo', chaninfo);
% 
% % 先绘制第一个头皮图
% figure;
% topoplot(restdata(:, 200), chanlocs, 'style', 'both', 'electrodes', 'labels', 'chaninfo', chaninfo);
% [s_eigvector, evr] = KLDfeatures_cal(taskdata);
% [task_red] = KLD_reduce(taskdata, num, s_eigvector);% 降维后的所有故障系统数据
% topoplot(alpha_data(:, t), channels32.chanlocs, 'style', 'both', 'electrodes', 'labels', 'chaninfo', channels32.chaninfo);
% 
% [s_eigvector, evr] = KLDfeatures_cal(restdata);
% [rest_red] = KLD_reduce(restdata, num, s_eigvector);% 降维后的所有故障系统数据
% topoplot(alpha_data(:, t), channels32.chanlocs, 'style', 'both', 'electrodes', 'labels', 'chaninfo', channels32.chaninfo);