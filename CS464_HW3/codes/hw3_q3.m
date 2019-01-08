function main(path)

path = strcat(path,'\digits.csv');
digits = load(path);

[coeff,score,latent] = pca(digits(:,1:end-1));
% Coeff is the principal component coefficients
% Latent is the eigen values of the covariance matrix

figure()
plot(latent);
ylabel('Eigen values of the covariance matrix');
xlabel('Index number of each Principal Component');

% Higher eigen values mean better Principal components
% The plot reveals that the PCA coefficients are already in order

figure();
for i = 1:5
    subplot(2,3,i)
    I = coeff( : , i);
    imagesc( reshape( I, 20, 20 ) ); 
    colormap( gray );
    axis image;
end
subplot(2,3,2)
title('Principal components with highest eigen values')

figure();
for i = 396:400
    subplot(2,3,i-395)
    I = coeff( :, i );
    imagesc( reshape( I, 20, 20 ) );
    colormap( gray );
    axis image;
end
subplot(2,3,2)
title('Principal components with lowest eigen values')

end