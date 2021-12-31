clc; clear;

% Cargamos los datos y los ajustamos
data = csvread('Clasificación_Vinos.csv');

% Hacemos la normalización de los datos
min = min(data);
max = max(data);
data = rescale(data,'InputMin',min,'InputMax',max);

[coeff,score,latent,tsquared,explained,mu]  = pca(data);

scatter(score(:,1), score(:,2));
figure();
scatter3(score(:,1), score(:,2), score(:,3));

header = {'X', 'Y'};
csvwrite_with_headers('vinos_pca.csv', score(:,1:2), header);



