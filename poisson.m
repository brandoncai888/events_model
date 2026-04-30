% Define the list of lambda values you want to compare
lambdas = [5, 4.5, 4.9]; % Add or remove numbers here
counts_upper = 30; 
time_lower = -4;
time_upper = 2;

% Set up the figure
figure('Name', 'Poisson Process Distributions Comparison', 'Position', [50, 50, 800, 800]);

% Create an empty cell array to hold the legend labels
legend_labels = cell(1, length(lambdas));

% Loop through each lambda value
for i = 1:length(lambdas)
    lambda = lambdas(i);
    
    % Store the text for the legend
    legend_labels{i} = sprintf('\\lambda = %d', lambda);
    
    %% 1. Counts per Time (Poisson PMF)
    subplot(3, 1, 1);
    x_counts = 0:counts_upper; 
    y_counts = poisspdf(x_counts, lambda); 
    % We use a line with circle markers ('-o') instead of stem to keep the overlay clean
    plot(x_counts, y_counts, '-o', 'LineWidth', 1.5, 'MarkerSize', 4);
    hold on; % This tells MATLAB to overlay the next plot on top of this one
    
    %% 2. Time Between Events (t) (Exponential PDF on Semi-Log X)
    subplot(3, 1, 2);
    t_time = logspace(time_lower, time_upper, 1000); 
    y_time = exppdf(t_time, 1/lambda); 
    semilogx(t_time, y_time, 'LineWidth', 2);
    hold on;
    
    %% 3. Frequency of Events (f = 1/t) PDF on Semi-Log X
    subplot(3, 1, 3);
    f_freq = logspace(-time_upper, -time_lower, 1000); 
    y_freq = (lambda ./ (f_freq.^2)) .* exp(-lambda ./ f_freq);
    semilogx(f_freq, y_freq, 'LineWidth', 2);
    hold on;
end

% --- Apply formatting and legends AFTER the loop is finished ---

subplot(3, 1, 1);
title('Counts per Time (Linear Scale)');
xlabel('Number of Events');
ylabel('Probability Mass');
legend(legend_labels, 'Location', 'best');
grid on;

subplot(3, 1, 2);
title('Time Between Events (Log X-Axis)');
xlabel('Time (t) [Log Scale]');
ylabel('Probability Density [Linear Scale]');
legend(legend_labels, 'Location', 'best');
grid on;

subplot(3, 1, 3);
title('Frequency, f = 1/t (Log X-Axis)');
xlabel('Frequency (f) [Log Scale]');
ylabel('Probability Density [Linear Scale]');
legend(legend_labels, 'Location', 'best');
grid on;

% Adjust layout
sgtitle('Comparing Poisson Process Distributions', 'FontSize', 14, 'FontWeight', 'bold');