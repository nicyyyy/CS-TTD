function [HSCimg,Phi] = CS_HS(HSimg,sub_band, Phi)
%UNTITLED8 此处显示有关此函数的摘要
%   HSimg = [m,n,band]
%   a   subrate
[m,n,band] = size(HSimg);
% sub_band = floor(band*a);

% rand_mat = randn(sub_band,band);
% min_v = min(min(rand_mat));
% max_v = max(max(rand_mat));
% normalized_mat = (rand_mat - min_v) / (max_v - min_v);

% Phi = sqrt(1/band) * normalized_mat;     % 感知矩阵（测量矩阵）   高斯随机矩阵

HSCimg = zeros(m,n,sub_band);
for i = 1:m
   for j = 1:n
       temp = HSimg(i,j,:);
       HSCimg(i,j,:) = (Phi * reshape(temp,[],band)');
   end
end
end

