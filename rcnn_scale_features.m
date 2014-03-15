function f = rcnn_scale_features(f, feat_norm_mean)
% My initial experiments were conducted on features with an average norm
% very close to 20. Using those features, I determined a good range of SVM
% C values to cross-validate over. Features from different layers end up
% have very different norms. We rescale all features to have an average norm
% of 20 (why 20? simply so that I can use the range of C values found in my 
% initial experiments), to make the same search range for C reasonable 
% regardless of whether these are pool5, fc6, or fc7 features. This strategy
% seems to work well. In practice, the optimal value for C ends up being the
% same across all features.
target_norm = 20;
f = f .* (target_norm / feat_norm_mean);
