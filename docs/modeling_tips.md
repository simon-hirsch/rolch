# Tips and Tricks

This page contains a collection of tips and tricks for using the online regression models.

## Help! My model fails to converge

If your distributional regression model does not converge, or fails to fit, you might want to check the following:

- Turn on the `verbose=3` option in the `Estimator()` class. This will print out the optimization steps and might give you a hint on what is going wrong.
- Turn on the `debug=True` option in the `Estimator()` class. This will save each iterations data to the estimator class. Remember to remove the option for production settings, otherwise the model size can increase significantly.
- Check the data for missing values.
- Check the data for features with zero variance. These cannot be handled by the `OnlineScaler()` and will cause missing / infinte values (due to the division by zero).
- Check the data for multicollinearity. The recursive least squares (RLS) algorithm uses the inverse of the Gramian $(X^T X)^{-1}$ which cannot be calculated if your $X$ matrix is not full rank. The algorithm will fail hard on this. The online coordinate descent (OCD) under the hood of the `LassoPathMethod` is more robust, since it does not rely on updating the *inverted* Gramian, but still will select arbitrary features in the case of multicollinearity. If LASSO works, but OLS not, there is a good chance that you have multicollinearity in your data.
- Is the distribution you're imposing on the data appropriate? Likelihood based methods fail miserably if the distribution is not appropriate. This especially concerns heavy tails and skewness, and potentially distributions that only live on the positive side of the real line.

If you have answered all of the above questions and the model still does not converge, [please open an issue with a reproducible example](https://github.com/simon-hirsch/rolch/issues). We will try to help you as best as we can.

## Scaling

Keep in mind which variables you'd like to scale, especially if you use lagged instances of the target variable. As the target variable $y$ is not scaled in the models, but all variables in $X$ are scaled (by default), the kind of 1:1 relationship you usually expect with lagged instances gets impacted. In the worst case, if severe distribution shift occurs in $y$, the model might starkly drift in an online prediction setting, since $L^l(y)$ are scaled by increasing/decreasing variance of $y$, but $y$ itself is not. 