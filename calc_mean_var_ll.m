function [mean_y_star, var_y_star, ll] = calc_mean_var_ll(K, K_star, K_star_star, y, i_train, i_test)

y_train = y(i_train);
n_train = sum(i_train);
n_test = sum(i_test);
L = chol(K,'lower');
alpha = L' \ (L \ y_train);

mean_y_star = K_star * alpha;

var_y_star = K_star_star;
for i = 1:n_test
    v = L \ K_star(i,:)';
    var_y_star(i,1) = var_y_star(i,1) - v' * v;
end

sum_log = 0;
for i = 1:n_train
    sum_log = sum_log + log(L(i,i));
end
ll = -0.5 * y_train' * alpha - sum_log - 0.5 * n_train * log(2 * pi);

end