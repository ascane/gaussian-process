function plot_prediction(y, mean_y_star, var_y_star, n_train_global, n_test_global, t_num_relative, i_test_global)

[y_filled, err_filled] = fill_mean_err(y, mean_y_star, var_y_star, n_train_global, n_test_global);
plot(t_num_relative(i_test_global), y_filled(i_test_global), 'r.');
% errorbar(t_num_relative, y_filled, err_filled,'r','MarkerEdgeColor','r','MarkerFaceColor','r');
fill([t_num_relative(i_test_global)' flipud(t_num_relative(i_test_global))'], [(y_filled(i_test_global)-err_filled(i_test_global))' (flipud(y_filled(i_test_global)+err_filled(i_test_global)))'],'r','linestyle','none');
alpha(0.25);

end