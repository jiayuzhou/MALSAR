function [ W, info ] = MTFLCd_SPGWrapper( X, y, lambda1, epsilon, opts )

opts.epsilon = epsilon;

[ W, info ] = MTFLCd_SPG( X, y, lambda1, 0, opts );

