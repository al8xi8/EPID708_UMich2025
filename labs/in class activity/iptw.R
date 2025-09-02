# ============================================================================
# IPTW Lab with Influence-Function Inference
# SSE 708: Machine Learning in the Era of Big Data
# Learning Objective: Apply IPTW estimators with realistic confounding
# ============================================================================

# --- package setup ----------------------------------------------------------
if (!require(SuperLearner)) install.packages("SuperLearner")
if (!require(randomForest)) install.packages("randomForest")
if (!require(ggplot2))      install.packages("ggplot2")
if (!require(dplyr))        install.packages("dplyr")

library(SuperLearner)
library(randomForest)
library(ggplot2)
library(dplyr)

cat("=== IPTW WITH INFLUENCE-FUNCTION SE LAB ===\n")
cat("Point Treatment Analysis with Inverse Probability Weighting\n\n")

# ============================================================================
# PART 1: SIMULATE REALISTIC PATIENT DATA
# ============================================================================

cat("PART 1: Creating Realistic Patient Population\n")
cat("==============================================\n\n")

generate_data <- function(n = 1000, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  age <- pmin(pmax(round(rnorm(n, 68, 12)), 30), 90)
  nyha <- sample(1:4, n, TRUE, prob = c(0.1, 0.4, 0.4, 0.1))
  ef   <- pmin(pmax(round(rnorm(n, 35, 10)), 15), 60)
  bnp  <- exp(rnorm(n, 6 + 0.15 * nyha - 0.02 * ef + 0.01 * age, 0.4))
  diabetes <- rbinom(n, 1, plogis(-1 + 0.02 * age + 0.3 * nyha))
  ckd      <- rbinom(n, 1, plogis(-2 + 0.03 * age + 0.2 * nyha))
  
  logit_ps <- -1.5 + 0.02 * age + 0.4 * nyha - 0.01 * ef +
    0.2 * log(bnp / 400) + 0.3 * diabetes + 0.4 * ckd
  ps       <- plogis(logit_ps)
  A        <- rbinom(n, 1, ps)
  
  logit_y  <- -2.5 + 0.03 * age + 0.6 * nyha - 0.015 * ef +
    0.3 * log(bnp / 400) + 0.4 * diabetes + 0.3 * ckd -
    1.0 * A                                    # strong effect
  py        <- plogis(logit_y)
  Y         <- rbinom(n, 1, py)
  
  data.frame(patient_id = 1:n, age, nyha, ef, bnp,
             diabetes, ckd, treatment = A, outcome = Y,
             true_prob = py, true_ps = ps)
}

set.seed(12345)
main_data <- generate_data(1000)

cat("Dataset characteristics:\n")
cat("- Sample size:", nrow(main_data), "\n")
cat("- Age mean (range):", round(mean(main_data$age),1), 
    "(",min(main_data$age),"-",max(main_data$age),")\n")
cat("- NYHA distribution:", paste0(names(table(main_data$nyha)),
                                   "=", as.numeric(table(main_data$nyha)), collapse=", "), "\n")
cat("- Treatment rate:", round(mean(main_data$treatment),3), "\n")
cat("- Outcome rate:",   round(mean(main_data$outcome),3), "\n\n")

treated_outcomes <- main_data %>% group_by(treatment) %>%
  summarise(outcome_rate = mean(outcome), .groups="drop")
cat("Naïve risk difference:",
    round(diff(treated_outcomes$outcome_rate),3), "\n\n")

# ============================================================================
# PART 2: TRUE (ORACLE) EFFECT
# ============================================================================

oracle <- (function(dat){
  logit1 <- with(dat, -2.5 + 0.03*age + 0.6*nyha - 0.015*ef +
                   0.3*log(bnp/400) + 0.4*diabetes + 0.3*ckd - 1.0)
  logit0 <- with(dat, -2.5 + 0.03*age + 0.6*nyha - 0.015*ef +
                   0.3*log(bnp/400) + 0.4*diabetes + 0.3*ckd)
  p1 <- plogis(logit1); p0 <- plogis(logit0)
  c(risk1 = mean(p1), risk0 = mean(p0), ate = mean(p1 - p0))
})(main_data)

cat("Oracle risk1 =", round(oracle["risk1"],4),
    "risk0 =", round(oracle["risk0"],4),
    "ATE =",    round(oracle["ate"],4), "\n\n")

# ============================================================================
# PART 3: IPTW WITH SUPERLEARNER
# ============================================================================

cat("PART 3: IPTW with SuperLearner\n")
cat("===============================\n\n")

# -- custom learner with basic interactions ----------------------------------
SL.glm.interaction <- function(Y, X, newX, family, ...) {
  int <- mutate(X,
                age_nyha = age*nyha,
                age_ef   = age*ef,
                nyha_ef  = nyha*ef,
                diabetes_ckd = diabetes*ckd)
  fit <- glm(Y ~ ., data = int, family = family)
  
  newXint <- mutate(newX,
                    age_nyha = age*nyha,
                    age_ef   = age*ef,
                    nyha_ef  = nyha*ef,
                    diabetes_ckd = diabetes*ckd)
  pred <- predict(fit, newdata = newXint, type = "response")
  out  <- list(pred = pred, fit = fit)
  class(out$fit) <- "SL.glm.interaction"
  out
}
predict.SL.glm.interaction <- function(object, newdata, ...){
  newdata <- mutate(newdata,
                    age_nyha = age*nyha,
                    age_ef   = age*ef,
                    nyha_ef  = nyha*ef,
                    diabetes_ckd = diabetes*ckd)
  predict(object, newdata = newdata, type="response")
}

iptw_analysis <- function(dat){
  
  X <- dat[,c("age","nyha","ef","bnp","diabetes","ckd")]
  A <- dat$treatment
  Y <- dat$outcome
  
  sl_lib <- c("SL.glm","SL.glm.interaction","SL.randomForest","SL.mean")
  
  ps_fit <- SuperLearner(Y = A, X = X, SL.library = sl_lib, family = binomial())
  ps     <- predict(ps_fit)$pred
  
  ps_tr  <- pmax(pmin(ps, .95), .05)
  wts    <- ifelse(A==1, 1/ps_tr, 1/(1-ps_tr))
  
  mu1 <- weighted.mean(Y[A==1], wts[A==1])
  mu0 <- weighted.mean(Y[A==0], wts[A==0])
  ate <- mu1 - mu0
  
  list(ate = ate, risk_treated = mu1, risk_control = mu0,
       ps = ps_tr, weights = wts, model = ps_fit)
}

iptw_results <- iptw_analysis(main_data)

cat("IPTW results: risk1 =", round(iptw_results$risk_treated,4),
    "risk0 =", round(iptw_results$risk_control,4),
    "ATE =",   round(iptw_results$ate,4), "\n\n")

# ============================================================================
# PART 4: INFLUENCE-FUNCTION STANDARD ERROR & 95% CI
# ============================================================================

cat("PART 4: Wald CI via Influence Function\n")
cat("======================================\n\n")

A   <- main_data$treatment
Y   <- main_data$outcome
ps  <- iptw_results$ps
n   <- nrow(main_data)
mu1 <- iptw_results$risk_treated
mu0 <- iptw_results$risk_control
ate <- iptw_results$ate

IF <-  A/ps        * (Y - mu1) -
  (1-A)/(1-ps) * (Y - mu0) +
  (mu1 - mu0) - ate       # centered

se_if <- sqrt( var(IF) / n )
z     <- qnorm(0.975)
ci_if <- ate + c(-1,1)*z*se_if

cat("Influence-function SE :", round(se_if, 4), "\n")
cat("95% Wald CI          :", round(ci_if[1],4), "to", round(ci_if[2],4), "\n")
cat("CI covers oracle?    :", oracle["ate"] >= ci_if[1] &
      oracle["ate"] <= ci_if[2], "\n\n")

# ============================================================================
# PART 5: VALIDATION SUMMARY
# ============================================================================

summary_table <- data.frame(
  Method  = c("Naïve", "IPTW"),
  Estimate= c(diff(treated_outcomes$outcome_rate), ate),
  SE      = c(NA, se_if),
  CI_Lower= c(NA, ci_if[1]),
  CI_Upper= c(NA, ci_if[2]),
  Bias    = c(diff(treated_outcomes$outcome_rate) - oracle["ate"],
              ate - oracle["ate"])
)

print(summary_table)
cat("\nOracle ATE:", round(oracle["ate"],4), "\n\n")

# ============================================================================
# PART 6: VISUALISATIONS
# ============================================================================

par(mfrow=c(1,2))

## (1) Propensity score distribution
plot(density(iptw_results$ps[main_data$treatment==0]),
     main="Propensity Score Distribution", xlab="PS", col="red", lwd=2, xlim=c(0,1))
lines(density(iptw_results$ps[main_data$treatment==1]), col="blue", lwd=2)
legend("topright",c("Control","Treated"),col=c("red","blue"),lty=1,lwd=2)

## (2) Bias comparison
barplot(summary_table$Bias,
        names.arg = summary_table$Method,
        col=c("red","lightblue"),
        main="Bias (Estimate - Oracle)", ylab="Risk difference")
abline(h=0,lwd=2)

par(mfrow=c(1,1))

# ============================================================================
# DISCUSSION PROMPTS (printed at end for students)
# ============================================================================

cat("=== DISCUSSION QUESTIONS ===\n")
cat("1. Compare the Wald CI width with what you observed from bootstrapping.\n")
cat("2. Why does the IF-based SE work even without re-sampling?\n")
cat("3. How would extreme weights influence both SE and bias?\n")
cat("4. Repeat with a smaller sample size – does coverage worsen?\n")
cat("5. Try adding unmeasured confounding – what happens?\n")

cat("\n=== LAB COMPLETE ===\n")
