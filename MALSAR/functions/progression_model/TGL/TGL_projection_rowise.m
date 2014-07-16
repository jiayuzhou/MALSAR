function w = TGL_projection_rowise(v, lambda_3)
%% FUNCTION TGL_projection_rowise
%   projection of temporal group Lasso (l21 projection).
%
%% OBJECTIVE:
% argmin_w { 0.5 \|w - v\|_2^2
%          + lambda_3 * \|v\|_2 }
% This is a simple thresholding:
%    w = max(\|v\|_2 - \lambda_3, 0)/\|v\|_2 * v
nm = norm(v, 2);
if nm == 0
    w = zeros(size(v));
else
    w = max(nm - lambda_3, 0)/nm * v;
end

end