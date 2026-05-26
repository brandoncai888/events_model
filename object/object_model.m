% MATLAB Script: First Passage Time Simulation vs Inverse Gaussian
clear; clc;

%% 1. Parameter Setup
show_expected = false; % <-- TOGGLE: Set to true to show theoretical IG curve, false to hide

a = 1000;          % Upward trend slope
sigma = 0.1;       % Noise scale (chosen small relative to x)
theta = 1;         % Mean reversion speed of the internal clock
x = 7;             % Threshold (notice x >> sigma)

num_sims = 10000;   % Number of random paths to simulate
dt = 0.000001;      % Time step size
max_t = 2.5 * (x/a); % Safety cap for max time

crossing_times = zeros(num_sims, 1);

%% 2. Monte Carlo Simulation Loop
fprintf('Running %d simulations...\n', num_sims);
tic;
for i = 1:num_sims
    t = 0;
    
    % At t=0, e^(0)=1, so W_1 ~ N(0,1). 
    % The stationary OU starting value is therefore Normal(0, sigma^2)
    Y = sigma * randn(); 
    X = a*t + Y;
    
    % Propagate until the process hits or exceeds threshold x
    while X < x && t < max_t
        t = t + dt;
        
        % Exact discrete time transition for a stationary OU process
        noise = randn();
        Y = Y * exp(-theta*dt) + sigma * sqrt(1 - exp(-2*theta*dt)) * noise;
        
        X = a*t + Y;
    end
    crossing_times(i) = t;
end
toc;

% Filter out any paths that didn't cross within max_t (highly unlikely for x >> sigma)
crossing_times(crossing_times >= max_t) = [];

%% 3. Analytical Inverse Gaussian Curve Calculation (Toggled)
if show_expected
    % Generate a fine time grid across the span of crossing times
    t_min = min(crossing_times);
    t_max = max(crossing_times);
    t_grid = linspace(t_min, t_max, 1000);
    
    % Corrected effective local variance for this specific time-warped process
    local_var = 2 * theta * sigma^2;
    
    % Inverse Gaussian (Wald) PDF formula
    pdf_ig = (x ./ sqrt(2 * pi * local_var .* t_grid.^3)) .* exp(-((x - a .* t_grid).^2) ./ (2 * local_var .* t_grid));
    
    % Calculate theoretical metrics
    mean_det = x / a;
    [~, idx] = max(pdf_ig);
    mode_ig = t_grid(idx);
end

%% 4. Plotting and Comparison
figure('Color', 'w');
hold on;

% Plot empirical histogram normalized to form a probability density function
histogram(crossing_times, 'Normalization', 'pdf', ...
          'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.7);

legend_entries = {'Simulated Paths'};

% Conditionally plot the theoretical lines if toggle is on
if show_expected
    plot(t_grid, pdf_ig, 'r-', 'LineWidth', 2.5);
    xline(mean_det, 'k--', 'LineWidth', 1.5);
    xline(mode_ig, 'g-.', 'LineWidth', 1.5);
    legend_entries = [legend_entries, {'Inverse Gaussian Model', 'Deterministic Center (x/a)', 'IG Mode (Peak)'}];
end

% Labels and Aesthetics
title(sprintf('First Crossing Time Distribution (x = %.1f, \\sigma = %.2f, \\theta = %.1f)', x, sigma, theta), 'FontSize', 12);
xlabel('First Passage Time (\tau)', 'FontSize', 11);
ylabel('Probability Density', 'FontSize', 11);
legend(legend_entries, 'Location', 'best');
grid on;
box on;
hold off;

%% 5. Verification Printout
fprintf('\n--- Results Verification ---\n');
fprintf('Empirical Mean Crossing Time:    %.6f\n', mean(crossing_times));

if show_expected
    fprintf('Deterministic Target Time (x/a): %.6f\n', mean_det);
    fprintf('Theoretical Curve Peak (Mode):   %.6f\n', mode_ig);
end