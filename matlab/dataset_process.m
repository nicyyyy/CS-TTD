%%
% meta learning dataset process
%%
% fname = 'abu-beach-1';
fname = 'Sandiego';
HSimg_mat = load(['./data_original/', fname, '.mat']);
HS_img = double(HSimg_mat.data);
map = double(HSimg_mat.map);
s = size(HS_img);
%%
% 提取目标区域
[tx,ty] = find(map == 1);
t_num = size(tx,1);
t = zeros(t_num, s(3));  %目标区域的光谱
for i=1:t_num
    temp = HS_img(tx(i),ty(i),:);
    t(i,:) = reshape(temp, [1, s(3)]);
%     ss = size(t);
end
prior = t;
save(['../data/prior/',fname,'-prior', '.mat'], 'prior')
%%
sub_band = 5;
Phi = load(['./phi/Phi-', num2str(sub_band), '/Phi_', num2str(s(3)), '_', num2str(sub_band), '.mat']);
Phi = Phi.Phi;
[data, phi] = CS_HS(HS_img, sub_band, Phi);

save(['../data/sub-band-',num2str(sub_band),'/',fname,'-',num2str(sub_band), '.mat'], 'data', 'map')