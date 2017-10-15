function [K, K_star, K_star_star] = kernels(k, t_num, y, i_train, i_test)

n_train = sum(i_train);
n_test = sum(i_test);

%K(i,j) := k(t_train_i, t_train_j)
K = zeros(n_train);
i_offset = 0;
for i = 1:n_train
    while i_train(i + i_offset) == 0
        i_offset = i_offset + 1;
    end
    j_offset = 0;
    for j = 1:n_train
        while i_train(j + j_offset) == 0
            j_offset = j_offset + 1;
        end
        K(i,j) = k(t_num(i + i_offset),t_num(j + j_offset));
    end
end

% K_star(i,j) := k(t_test_i, t_train_j)
K_star = zeros(n_test,n_train);
i_offset = 0;
for i = 1:n_test
    while i_test(i + i_offset) == 0
        i_offset = i_offset + 1;
    end
    j_offset = 0;
    for j = 1:n_train
        while i_train(j + j_offset) == 0
            j_offset = j_offset + 1;
        end
        K_star(i,j) = k(t_num(i + i_offset),t_num(j + j_offset));
    end
end

% K_star_star(i,1) := k(t_test_i, t_test_i)
K_star_star = zeros(n_test,1);
i_offset = 0;
for i = 1:n_test
    while i_test(i + i_offset) == 0
        i_offset = i_offset + 1;
    end
    K_star_star(i,1) = k(t_num(i + i_offset),t_num(i + i_offset));
end

end