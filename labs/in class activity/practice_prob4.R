#############################################################
#  Practice-problem 4  •  Marginal Structural Model (MSM)
#  ---------------------------------------------------------
#  – Binary outcome  Y
#  – Discrete dose   A ∈ {0,1,2,3}
#  – Two baseline covariates W1, W2
#  – Flexible outcome model = random forest
#############################################################

# -------- 0.  Packages -------------------------------------------------------
# install.packages(c("randomForest", "ggplot2", "dplyr"))  # once only
library(randomForest)   # flexible learner
library(ggplot2)        # plots
library(dplyr)          # pipes / mutate

set.seed(123)           # reproducible

# -------- 1.  Simulate a data set -------------------------------------------
n  <- 4000                          # sample size (large to reduce noise)

W1 <- rbinom(n, 1, 0.5)             # binary risk factor
W2 <- rnorm(n, 0, 1)                # continuous risk factor
A  <- sample(0:3, n, replace = TRUE)  # exposure dose 0-3

# True probability  P(Y=1 | A, W)
logit_p <- -2            +  1.4*A      - 0.3*A^2 +   # quadratic in A
  1.0*W1        +  0.8*W2                    # baseline risk
pY       <- 1 / (1 + exp(-logit_p))
Y        <- rbinom(n, 1, pY)

dat <- data.frame(Y = factor(Y), A, W1, W2)

# -------- 2.  Fit a flexible outcome model ----------------------------------
rf_fit <- randomForest(Y ~ A + W1 + W2, data = dat, ntree = 50)

# -------- 3.  Counterfactual mean for each a = 0,1,2,3 ----------------------
a_grid <- 0:3

cf_means <- sapply(a_grid, function(a_val) {
  # copy original data but set EVERYONE's A = a_val
  tmp <- dat
  tmp$A <- a_val
  # predict P(Y=1) for each person, then average
  mean( predict(rf_fit, newdata = tmp, type = "prob")[, "1"] )
})

cf_tbl <- data.frame(A = a_grid, EY = cf_means)

# -------- 4.  Fit the quadratic MSM  E[Yᵃ] = β0 + β1 a + β2 a² --------------
msm_fit <- lm(EY ~ A + I(A^2), data = cf_tbl)
coef(msm_fit)

# -------- 5.  Plot: dots = cf means, curve = fitted MSM ---------------------
ggplot(cf_tbl, aes(A, EY)) +
  geom_point(size = 3, colour = "steelblue") +
  stat_function(fun = function(x)
    coef(msm_fit)[1] + coef(msm_fit)[2]*x + coef(msm_fit)[3]*x^2,
    linetype = "dashed") +
  labs(title = "Marginal mean curve  E[Yᵃ] vs exposure a",
       x = "Dose a",
       y = "Predicted probability Y=1") +
  theme_minimal(base_size = 12)

# -------- 6.  Read off the ATE for  a = 3  vs  a = 0 ------------------------
b  <- coef(msm_fit)                # b[1]=β0, b[2]=β1, b[3]=β2
ATE_3_0 <- (b[1] + b[2]*3 + b[3]*3^2) - (b[1] + b[2]*0 + b[3]*0^2)

cat(sprintf("\nEstimated E[Y^3] – E[Y^0]  =  %.4f\n", ATE_3_0))

# quick check: same result using shortcut β₁·3 + β₂·9
cat(sprintf("Shortcut with coefficients      =  %.4f\n",
            b[2]*3 + b[3]*9))

# -------- 7.  Causal attributable risk  -------------------------------------
#  AR = P(Y=1) – E[Y^{A=0}]
AR <- mean(as.numeric(as.character(dat$Y))) - cf_tbl$EY[cf_tbl$A == 0]
cat(sprintf("\nCausal attributable risk (AR)  =  %.4f\n", AR))
