% 1. Load the data, defining the tide height readings to be y and the
% reading times to be t.
filename = fullfile('C:','Users','ascan','Desktop','GPlab','sotonmet.txt'); % Please change the path
fileID = fopen(filename);
C_text = textscan(fileID,'%s',19,'Delimiter',',');
C_data = textscan(fileID,'%s %f %s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f','Delimiter',',');
fclose(fileID);
strtodatetime = @(x) datetime(x,'InputFormat','yyyy-MM-dd''T''HH:mm:ss');
t = strtodatetime(C_data{3}); % reading date and time
t_num_relative = datenum(t) - datenum(t(1));
y = C_data{6}; % tide height
i_train_global = ~isnan(y);
i_test_global = isnan(y);
n_train_global = sum(i_train_global);
n_test_global = sum(i_test_global);

% 2. Write your own Gaussian process code to perform retrospective
% prediction for the missing readings. Hold covariance and mean function
% hyperparameters fixed to sensible values of your choice as a first step.

% Exponentiated quadratic: k(t,t') := \sigma_f^2 \exp(\frac{-(t-t')^2}{2 l^2})
sigma_f = 1.25; % 1.25
l = 0.18; % 0.18
sigma_n = 0.2; % 0.2
kronDel = @(j, k) j==k ;
k_eq = @(t1,t2) sigma_f^2 * exp(-(t1-t2)^2/(2*l^2)) + sigma_n^2 * kronDel(t1,t2);
[K_eq, K_eq_star, K_eq_star_star] = kernels(k_eq, t_num_relative, y, i_train_global, i_test_global);
[mean_y_eq_star, var_y_eq_star, ll_eq] = calc_mean_var_ll(K_eq, K_eq_star, K_eq_star_star, y, i_train_global, i_test_global);

% Rational Quadratic
sigma_r = 1;
alpha = 0.1;
l_r = 0.18;
sigma_n = 0.1;
k_rq = @(t1,t2) sigma_r^2 * (1 + 1/(2 * alpha) * ((t1 - t2)/l_r)^2)^(-alpha) + sigma_n^2 * kronDel(t1,t2);
[K_rq, K_rq_star, K_rq_star_star] = kernels(k_rq, t_num_relative, y, i_train_global, i_test_global);
[mean_y_rq_star, var_y_rq_star, ll_rq] = calc_mean_var_ll(K_rq, K_rq_star, K_rq_star_star, y, i_train_global, i_test_global);

% Periodic
sigma_p = 1;
sigma_n = 0.2;
l_p = 0.2;
rho = 0.51;
k_p = @(t1,t2) sigma_p^2 * exp(-2*(sin(pi * (t1-t2) /rho))^2 / l_p) + sigma_n^2 * kronDel(t1,t2);
[K_p, K_p_star, K_p_star_star] = kernels(k_p, t_num_relative, y, i_train_global, i_test_global);
[mean_y_p_star, var_y_p_star, ll_p] = calc_mean_var_ll(K_p, K_p_star, K_p_star_star, y, i_train_global, i_test_global);

% Matern 3/2
sigma_m = 2.75;
sigma_n = 0.2;
l_m = 0.26;
k_m = @(t1,t2) sigma_m^2 * (1 + sqrt(3) * abs(t1-t2) / l_m) * exp(-sqrt(3) * abs(t1-t2) / l_m) + sigma_n^2 * kronDel(t1,t2);
[K_m, K_m_star, K_m_star_star] = kernels(k_m, t_num_relative, y, i_train_global, i_test_global);
[mean_y_m_star, var_y_m_star, ll_m] = calc_mean_var_ll(K_m, K_m_star, K_m_star_star, y, i_train_global, i_test_global);

% 3. Compare against the ground truth tide heights using root-mean-square
% -error or the predictive log-likelihood, log p(test data|training data).
y_true = C_data{11};
plot(t_num_relative, y_true, 'b.');
hold on;

[rms_eq] = rms_error(y_true, mean_y_eq_star, i_test_global);
plot_prediction(y, mean_y_eq_star, var_y_eq_star, n_train_global, n_test_global, t_num_relative, i_test_global);

% Title for exponentiated quadratic kernel.
kernel_str = texlabel('k(t_1,t_2)=sigma_f^2 exp(-(t_1-t_2)^2/(2l^2))+sigma_n^2 delta(t_1,t_2)');
title_str_1 = sprintf('Prediction of tide height using Gaussian Process\n Exponentiated quadratic kernel: ');
sigma_f_str_1 = texlabel('sigma_f');
sigma_f_str_2 = sprintf(' = %f,', sigma_f);
l_str = sprintf('l = %f,', l);
sigma_n_str_1 = texlabel('sigma_n');
sigma_n_str_2 = sprintf(' = %f. ', sigma_n);
rms_str = sprintf('rms = %f,', rms_eq);
ll_str = sprintf('log-likelihood = %f', ll_eq);
title(strcat(title_str_1, kernel_str, sprintf('.\n where '), sigma_f_str_1, sigma_f_str_2, l_str, sigma_n_str_1, sigma_n_str_2, rms_str, ll_str));

[rms_rq] = rms_error(y_true, mean_y_rq_star, i_test_global);
plot_prediction(y, mean_y_rq_star, var_y_rq_star, n_train_global, n_test_global, t_num_relative, i_test_global);
title(sprintf('sigma_r = %f, alpha = %f, l_r = %f, sigma_n = %f, rms = %f, ll = %f', sigma_r, alpha, l_r, sigma_n, rms_rq, ll_rq));

[rms_p] = rms_error(y_true, mean_y_p_star, i_test_global);
plot_prediction(y, mean_y_p_star, var_y_p_star, n_train_global, n_test_global, t_num_relative, i_test_global);
title(sprintf('sigma_p = %f, rho = %f, l_p = %f, sigma_n = %f, rms = %f, ll = %f', sigma_p, rho, l_p, sigma_n, rms_p, ll_p));

[rms_m] = rms_error(y_true, mean_y_m_star, i_test_global);
plot_prediction(y, mean_y_m_star, var_y_m_star, n_train_global, n_test_global, t_num_relative, i_test_global);
title(sprintf('sigma_m = %f, l_m = %f, sigma_n = %f, rms = %f, ll = %f', sigma_m, l_m, sigma_n, rms_m, ll_m));

xlabel('relative time (days)');
ylabel('tide height (m)');
lgnd = legend('ground truth','prediction', '+/- 1.96 SD', 'Location','southoutside');
set(lgnd, 'color', 'none');
axis([0 t_num_relative(end) 0 6]);

% 4. Test some more sophisticated covariance and mean functions to try to
% improve performance.

% 5. Investigate alternate means of managing the hyperparameters, including
% maximum likelihood and maximum a-posteriori.

% 6. Produce code to allow for sequential prediction: that is, using only
% readings from prior to (and including) a time t to predict for the
% readings at t. This might seem a bit easy, as you have an observation at
% time t, but the noise in the observations means that it's still
% interesting to do.

lookahead = 1;
i_offset = 0;
mean_y_eq_star_seq = zeros(n_test_global,1);
var_y_eq_star_seq = zeros(n_test_global,1);
for i = 1:n_test_global
    while i_test_global(i + i_offset) == 0
        i_offset = i_offset + 1;
    end
    i_current = i + i_offset
    % Find the indices of training data for sequential prediction.
    i_train_seq = i_train_global;
    for ii = 1:n_train_global + n_test_global
        if (t_num_relative(ii,1) + lookahead < t_num_relative(i_current,1)) || (ii >= i_current)
            i_train_seq(ii,1) = 0;
        end
    end
    i_test_seq = zeros(n_train_global + n_test_global, 1);
    i_test_seq(i_current, 1) = 1;
    [K_eq_seq, K_eq_star_seq, K_eq_star_star_seq] = kernels(k_eq, t_num_relative, y, i_train_seq, i_test_seq);
    [mean_y_eq_star_seq_temp, var_y_eq_star_seq_temp, ll_eq_seq] = calc_mean_var_ll(K_eq_seq, K_eq_star_seq, K_eq_star_star_seq, y, i_train_seq, i_test_seq);
    mean_y_eq_star_seq(i,1) = mean_y_eq_star_seq_temp;
    var_y_eq_star_seq(i,1) = var_y_eq_star_seq_temp;
end

% 7. Create plots that show such predictions for a fixed lookahead (the
% separation in time between the most recent reading and the time at which
% predictions are made in the sequential setting). That is, such plots
% should show predictions for t + lookahead given data only prior to (and
% including) t, for all t.

[rms_err_seq] = rms_error(y_true, mean_y_eq_star_seq, i_test_global);
plot_prediction(y, mean_y_eq_star_seq, var_y_eq_star_seq, n_train_global, n_test_global, t_num_relative, i_test_global);

% 8. Investigate the impact of increasing the lookahead. Results for this
% kind of experiment are plotted in Figure 1.

% 9. Repeat experiments with other readings available in the data, starting
% with air temperature.

% 10. Investigate the possibility of fusing multiple readings: e.g. using
% both air temperature and tide height within a single model to improve
% performance.
