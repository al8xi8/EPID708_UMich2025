# =====================================================================
#  TMLE LAB · SSE 708
#  Point-treatment causal effect (flexible learners) + weight read-out
# =====================================================================

## ------------------ package setup -----------------------------------
pkgs <- c("SuperLearner", "xgboost", "glmnet",
          "randomForest", "ggplot2", "dplyr")
need <- pkgs[!pkgs %in% installed.packages()[,1]]
if (length(need))
  install.packages(need, repos = "https://cloud.r-project.org")
invisible(lapply(pkgs, library, character.only = TRUE))

cat("=== TMLE LAB (XGBoost · RF · glmnet) ===\n\n")

# =====================================================================
#  PART 1 · simulate realistic data
# =====================================================================
generate_data <- function(n = 1000, seed = 1) {
  set.seed(seed)
  age <- pmin(pmax(round(rnorm(n, 68, 12)), 30), 90)
  nyha <- sample(1:4, n, TRUE, prob = c(0.1, 0.4, 0.4, 0.1))
  ef   <- pmin(pmax(round(rnorm(n, 35, 10)), 15), 60)
  bnp  <- exp(rnorm(n, 6 + 0.15*nyha - 0.02*ef + 0.01*age, 0.4))
  diabetes <- rbinom(n, 1, plogis(-1 + 0.02*age + 0.3*nyha))
  ckd      <- rbinom(n, 1, plogis(-2 + 0.03*age + 0.2*nyha))
  
  logit_ps <- -1.5 + 0.02*age + 0.4*nyha - 0.01*ef +
    0.2*log(bnp/400) + 0.3*diabetes + 0.4*ckd
  ps <- plogis(logit_ps)
  A  <- rbinom(n, 1, ps)
  
  logit_y <- -2.5 + 0.03*age + 0.6*nyha - 0.015*ef +
    0.3*log(bnp/400) + 0.4*diabetes + 0.3*ckd - 1*A
  py <- plogis(logit_y)
  Y  <- rbinom(n, 1, py)
  
  data.frame(age, nyha, ef, bnp, diabetes, ckd,
             treatment = A, outcome = Y,
             true_ps = ps, true_prob = py)
}

dat <- generate_data(1000)

treated_outcomes <- dat |> group_by(treatment) |>
  summarise(risk = mean(outcome), .groups = "drop")
cat("Naïve risk diff:", round(diff(treated_outcomes$risk), 4), "\n\n")

# =====================================================================
#  PART 2 · oracle truth
# =====================================================================
oracle <- with(dat, {
  logit1 <- -2.5 + 0.03*age + 0.6*nyha - 0.015*ef +
    0.3*log(bnp/400) + 0.4*diabetes + 0.3*ckd - 1
  logit0 <- logit1 + 1
  mean(plogis(logit1) - plogis(logit0))
})
cat("Oracle ATE =", round(oracle, 4), "\n\n")

# =====================================================================
#  PART 3 · manual TMLE (with weight print-outs)
# =====================================================================
tmle_manual <- function(df) {
  
  X <- df[, c("age","nyha","ef","bnp","diabetes","ckd")]
  A <- df$treatment
  Y <- df$outcome
  n <- nrow(df)
  
  SL.lib <- c("SL.glmnet", "SL.xgboost", "SL.randomForest", "SL.mean")
  
  # ---------- outcome Super Learner (Q) ------------------------------
  Q_fit <- SuperLearner(Y, data.frame(A, X),
                        family = binomial(), SL.library = SL.lib)
  
  cat("\n--- Super Learner weights for Q (outcome) ---\n")
  print(round(Q_fit$coef, 3))
  
  Q0n <- predict(Q_fit, newdata = data.frame(A = 0, X))$pred
  Q1n <- predict(Q_fit, newdata = data.frame(A = 1, X))$pred
  QAn <- ifelse(A == 1, Q1n, Q0n)
  
  # ---------- propensity Super Learner (g) ---------------------------
  g_fit <- SuperLearner(A, X, family = binomial(), SL.library = SL.lib)
  
  cat("\n--- Super Learner weights for g (propensity) ---\n")
  print(round(g_fit$coef, 3))
  
  gn1 <- pmin(pmax(predict(g_fit)$pred, 0.01), 0.99)
  gn0 <- 1 - gn1
  
  # ---------- clever covariate & fluctuation ------------------------
  H     <- A/gn1 - (1-A)/gn0
  logit <- function(p) log(p/(1-p))
  eps   <- coef(glm(Y ~ -1 + H + offset(logit(QAn)),
                    family = binomial()))
  Q1s <- plogis(logit(Q1n) + eps/gn1)
  Q0s <- plogis(logit(Q0n) - eps/gn0)
  
  psi <- mean(Q1s - Q0s)
  IC  <- A/gn1*(Y - Q1s) - (1-A)/gn0*(Y - Q0s) + (Q1s - Q0s) - psi
  se  <- sqrt(var(IC)/n)
  ci  <- psi + c(-1,1) * qnorm(.975) * se
  
  list(ate = psi, se = se, ci = ci)
}

tmle_res <- tmle_manual(dat)
cat("\nTMLE ATE =", round(tmle_res$ate,4),
    "| SE =", round(tmle_res$se,4),
    "| 95% CI:", paste(round(tmle_res$ci,4), collapse=" – "), "\n\n")

# =====================================================================
#  PART 4 · summary & visuals
# =====================================================================
naive_rd <- diff(treated_outcomes$risk)
summary_tbl <- tibble(
  Method   = c("Naïve","TMLE"),
  Estimate = c(naive_rd, tmle_res$ate),
  SE       = c(NA, tmle_res$se),
  CI_Lower = c(NA, tmle_res$ci[1]),
  CI_Upper = c(NA, tmle_res$ci[2]),
  Bias     = c(naive_rd - oracle, tmle_res$ate - oracle)
)
print(summary_tbl)

par(mfrow = c(1,2))
barplot(summary_tbl$Bias, names.arg = summary_tbl$Method,
        col = c("orange","lightgreen"),
        main = "Bias vs Oracle", ylab = "Risk diff")
abline(h = 0, lwd = 2)

hist(tmle_res$ate + rnorm(1000, 0, tmle_res$se),
     breaks = 20, main = "Sampling dist. (simulated)",
     xlab = "ATE", col = "steelblue")
abline(v = tmle_res$ci, col = "red", lwd = 2)
par(mfrow = c(1,1))

# =====================================================================
#  PART 5 · discussion prompts
# =====================================================================
cat("=== DISCUSSION PROMPTS ===\n")
cat("1.  Compare TMLE's CI with a bootstrap CI.\n")
cat("2.  Explain why H = A/g - (1-A)/(1-g) is an efficient score.\n")
cat("3.  What happens if you truncate PS at 0.20 instead of 0.01?\n")
cat("4.  Replace Super Learner with a mis-specified GLM—does TMLE still\n",
    "    beat the naïve estimator?\n")
cat("5.  How would poor positivity show up in gn1 and IC diagnostics?\n")
cat("\n=== LAB COMPLETE ===\n")
