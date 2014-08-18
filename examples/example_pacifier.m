% test the main formulation 

addpath('../MALSAR/c_files/largescale_ops/');
addpath('../MALSAR/functions/pacifier/');

clear; clc;
rng(1985)
n = 12;
k = 40;
d = 1000;
t_min = 150;
t_max = 200;
density = 0.001;

X0 = cell(n, 1); 
for i = 1 : n
    X0{i} = sprand(d, t_min + randi(t_max-t_min, 1), density);
end

reg_l1 = 1e-10;
reg_l2 = 1e-10;

reg_smooth = 10;

[ U1, V1, Ss1, fv1 ] = pacifier_iba( X0, k+5, reg_l1, reg_l2, reg_smooth);
[ U2, V2, Ss2, fv2 ] = pacifier_sba( X0, k+5, reg_l1, reg_l2, reg_smooth);

figure
plot(fv1)
title('Pacifier IBA Objective Value')

figure
plot(fv2)
title('Pacifier SBA Objective Value')