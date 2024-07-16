q = double(RDP_Sample_quant);
w = t.L001_layers_0_self_attn_linear_v_weight;
w = double(w');

out = (q - 72)*w;

vt = double(linear_v_out) - 46;
% vt = reshape(vt, [1,64]);
round(out/256) - vt
% round(round(out/2048))*4 - vt


