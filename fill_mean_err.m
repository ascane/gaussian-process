function [y_filled, err_filled] = fill_mean_err(y, mean_y_star, var_y_star, n_train_global, n_test_global)

y_filled = y;
err_filled = zeros(n_train_global + n_test_global,1);
i_offset = 0;
for i = 1:n_test_global
    while ~isnan(y(i + i_offset))
        i_offset = i_offset + 1;
    end
    y_filled(i + i_offset,1) = mean_y_star(i,1);
    err_filled(i + i_offset,1) = 1.96 * sqrt(var_y_star(i,1));
end

end