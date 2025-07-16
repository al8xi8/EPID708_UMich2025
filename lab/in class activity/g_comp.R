# ============================================================================
# G-Computation with Bootstrap Lab: Point Treatment Analysis
# SSE 708: Machine Learning in the Era of Big Data
# Learning Objective: Apply substitution estimators with realistic confounding
# ============================================================================

# Load required packages
if (!require(SuperLearner)) install.packages("SuperLearner")
if (!require(randomForest)) install.packages("randomForest")
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(dplyr)) install.packages("dplyr")

library(SuperLearner)
library(randomForest)
library(ggplot2)
library(dplyr)

cat("=== G-COMPUTATION WITH BOOTSTRAP LAB ===\n")
cat("Point Treatment Analysis with Realistic Confounding\n\n")

# ============================================================================
# PART 1: SIMULATE REALISTIC PATIENT DATA
# ============================================================================

cat("PART 1: Creating Realistic Patient Population\n")
cat("==============================================\n\n")

# Function to generate one dataset
generate_data <- function(n = 1000, seed = NULL) {
  if(!is.null(seed)) set.seed(seed)
  
  # Baseline patient characteristics (heart failure patients)
  age <- round(rnorm(n, mean = 68, sd = 12))
  age <- pmax(age, 30)  # Minimum age 30
  age <- pmin(age, 90)  # Maximum age 90
  
  # NYHA class (heart failure severity: I=mild, IV=severe)
  nyha_prob <- c(0.1, 0.4, 0.4, 0.1)  # Mostly class II-III
  nyha <- sample(1:4, n, replace = TRUE, prob = nyha_prob)
  
  # Ejection fraction (heart pumping efficiency)
  ef <- round(rnorm(n, mean = 35, sd = 10))
  ef <- pmax(ef, 15)  # Minimum EF
  ef <- pmin(ef, 60)  # Maximum EF
  
  # Biomarker: BNP (higher = worse heart failure)
  bnp <- exp(rnorm(n, mean = 6 + 0.15*nyha - 0.02*ef + 0.01*age, sd = 0.4))
  
  # Comorbidities
  diabetes <- rbinom(n, 1, plogis(-1 + 0.02*age + 0.3*nyha))
  ckd <- rbinom(n, 1, plogis(-2 + 0.03*age + 0.2*nyha))
  
  # Treatment assignment (depends on confounders - this creates confounding!)
  # Sicker patients more likely to get treatment
  treatment_logit <- -1.5 + 0.02*age + 0.4*nyha - 0.01*ef + 
    0.2*log(bnp/400) + 0.3*diabetes + 0.4*ckd
  
  treatment_prob <- plogis(treatment_logit)
  treatment <- rbinom(n, 1, treatment_prob)
  
  # Outcome: 6-month heart failure hospitalization
  # True causal effect: treatment reduces risk (strong effect for clear lab results)
  outcome_logit <- -2.5 + 0.03*age + 0.6*nyha - 0.015*ef + 
    0.3*log(bnp/400) + 0.4*diabetes + 0.3*ckd +
    -1.0*treatment  # TRUE TREATMENT EFFECT = -1.0 on log-odds scale (OR ≈ 0.37)
  
  outcome_prob <- plogis(outcome_logit)
  outcome <- rbinom(n, 1, outcome_prob)
  
  # Create dataset
  data <- data.frame(
    patient_id = 1:n,
    age = age,
    nyha = nyha,
    ef = ef,
    bnp = bnp,
    diabetes = diabetes,
    ckd = ckd,
    treatment = treatment,
    outcome = outcome,
    # Store true probabilities for comparison
    true_prob = outcome_prob
  )
  
  return(data)
}

# Generate main dataset
set.seed(12345)
main_data <- generate_data(n = 1000)

cat("Dataset characteristics:\n")
cat("- Sample size:", nrow(main_data), "\n")
cat("- Age: mean =", round(mean(main_data$age), 1), 
    ", range =", min(main_data$age), "-", max(main_data$age), "\n")
cat("- NYHA distribution:", table(main_data$nyha), "\n")
cat("- Treatment rate:", round(mean(main_data$treatment), 3), "\n")
cat("- Outcome rate:", round(mean(main_data$outcome), 3), "\n\n")

# Show confounding
cat("Evidence of confounding:\n")
treated_outcomes <- main_data %>% group_by(treatment) %>% 
  summarise(outcome_rate = mean(outcome), .groups = 'drop')
print(treated_outcomes)
cat("Naive treatment effect:", round(diff(treated_outcomes$outcome_rate), 3), "\n\n")

# ============================================================================
# PART 2: CALCULATE THE TRUE TREATMENT EFFECT (ORACLE)
# ============================================================================

cat("PART 2: Calculate True Treatment Effect (Oracle Knowledge)\n")
cat("=========================================================\n\n")

# We know the true data-generating process, so we can calculate the true ATE
calculate_true_ate <- function(data) {
  # Counterfactual outcomes under treatment = 1
  outcome_logit_1 <- -2.5 + 0.03*data$age + 0.6*data$nyha - 0.015*data$ef + 
    0.3*log(data$bnp/400) + 0.4*data$diabetes + 0.3*data$ckd +
    -1.0*1  # Set treatment = 1 (using strong effect for lab)
  prob_1 <- plogis(outcome_logit_1)
  
  # Counterfactual outcomes under treatment = 0  
  outcome_logit_0 <- -2.5 + 0.03*data$age + 0.6*data$nyha - 0.015*data$ef + 
    0.3*log(data$bnp/400) + 0.4*data$diabetes + 0.3*data$ckd +
    -1.0*0  # Set treatment = 0
  prob_0 <- plogis(outcome_logit_0)
  
  # True ATE
  true_ate <- mean(prob_1) - mean(prob_0)
  
  return(list(
    ate = true_ate,
    risk_treated = mean(prob_1),
    risk_control = mean(prob_0)
  ))
}

true_effects <- calculate_true_ate(main_data)

cat("TRUE TREATMENT EFFECTS (Oracle knowledge):\n")
cat("- Risk under universal treatment:", round(true_effects$risk_treated, 4), "\n")
cat("- Risk under universal control:", round(true_effects$risk_control, 4), "\n")
cat("- True ATE (risk difference):", round(true_effects$ate, 4), "\n\n")

# ============================================================================
# PART 3: G-COMPUTATION WITH SUPERLEARNER
# ============================================================================

cat("PART 3: G-Computation with SuperLearner\n")
cat("=======================================\n\n")

# G-computation function
gcomputation <- function(data) {
  
  # Prepare data for SuperLearner
  # Features for outcome model
  X <- data[, c("age", "nyha", "ef", "bnp", "diabetes", "ckd", "treatment")]
  Y <- data$outcome
  
  # SuperLearner library
  SL.library <- c("SL.glm", "SL.randomForest", "SL.xgboost")
  
  cat("Fitting SuperLearner outcome model...\n")
  
  # Fit outcome model Q(A,W) = E[Y|A,W]
  outcome_model <- SuperLearner(Y = Y, X = X, 
                                SL.library = SL.library,
                                family = binomial())
  
  # Print model weights
  cat("SuperLearner weights:\n")
  print(round(outcome_model$coef, 3))
  cat("\n")
  
  # Predict under treatment = 1 for everyone
  X_treated <- X
  X_treated$treatment <- 1
  pred_treated <- predict(outcome_model, newdata = X_treated)$pred
  
  # Predict under treatment = 0 for everyone  
  X_control <- X
  X_control$treatment <- 0
  pred_control <- predict(outcome_model, newdata = X_control)$pred
  
  # Calculate ATE
  risk_treated <- mean(pred_treated)
  risk_control <- mean(pred_control)
  ate <- risk_treated - risk_control
  
  return(list(
    ate = ate,
    risk_treated = risk_treated,
    risk_control = risk_control,
    pred_treated = pred_treated,
    pred_control = pred_control,
    model = outcome_model
  ))
}

# Estimate using G-computation
gcomp_results <- gcomputation(main_data)

cat("G-COMPUTATION RESULTS:\n")
cat("- Estimated risk under treatment:", round(gcomp_results$risk_treated, 4), "\n")
cat("- Estimated risk under control:", round(gcomp_results$risk_control, 4), "\n")
cat("- Estimated ATE:", round(gcomp_results$ate, 4), "\n")
cat("- True ATE:", round(true_effects$ate, 4), "\n")
cat("- Bias:", round(gcomp_results$ate - true_effects$ate, 4), "\n\n")

# ============================================================================
# PART 4: BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

cat("PART 4: Bootstrap Inference\n")
cat("===========================\n\n")

# Bootstrap function
bootstrap_gcomp <- function(data, B = 100) {
  
  bootstrap_results <- data.frame(
    boot_sample = 1:B,
    ate = NA,
    risk_treated = NA,
    risk_control = NA
  )
  
  cat("Running", B, "bootstrap samples...\n")
  
  # Progress tracking
  progress_points <- round(seq(1, B, length.out = 10))
  
  for(b in 1:B) {
    if(b %in% progress_points) {
      cat("  Bootstrap sample", b, "of", B, "\n")
    }
    
    # Sample with replacement
    boot_indices <- sample(nrow(data), nrow(data), replace = TRUE)
    boot_data <- data[boot_indices, ]
    
    # Run G-computation on bootstrap sample
    tryCatch({
      boot_results <- gcomputation(boot_data)
      bootstrap_results$ate[b] <- boot_results$ate
      bootstrap_results$risk_treated[b] <- boot_results$risk_treated
      bootstrap_results$risk_control[b] <- boot_results$risk_control
    }, error = function(e) {
      # If SuperLearner fails, skip this bootstrap sample
      cat("Warning: Bootstrap sample", b, "failed\n")
    })
  }
  
  return(bootstrap_results)
}

# Run bootstrap
cat("Starting bootstrap procedure...\n")
set.seed(999)
bootstrap_results <- bootstrap_gcomp(main_data, B = 50)  # Smaller B for demo

# Remove failed bootstrap samples
bootstrap_results <- bootstrap_results[!is.na(bootstrap_results$ate), ]
n_successful <- nrow(bootstrap_results)

cat("Successful bootstrap samples:", n_successful, "\n\n")

# Calculate confidence intervals
if(n_successful >= 10) {
  ci_level <- 0.95
  alpha <- 1 - ci_level
  
  ate_ci <- quantile(bootstrap_results$ate, c(alpha/2, 1-alpha/2), na.rm = TRUE)
  
  # Bootstrap statistics
  bootstrap_mean <- mean(bootstrap_results$ate, na.rm = TRUE)
  bootstrap_se <- sd(bootstrap_results$ate, na.rm = TRUE)
  
  cat("BOOTSTRAP RESULTS:\n")
  cat("- Bootstrap mean ATE:", round(bootstrap_mean, 4), "\n")
  cat("- Bootstrap SE:", round(bootstrap_se, 4), "\n")
  cat("- 95% CI:", round(ate_ci[1], 4), "to", round(ate_ci[2], 4), "\n")
  
  # Check if CI covers truth
  covers_truth <- (true_effects$ate >= ate_ci[1]) & (true_effects$ate <= ate_ci[2])
  cat("- CI covers truth:", covers_truth, "\n\n")
  
} else {
  cat("Not enough successful bootstrap samples for CI calculation\n\n")
}

# ============================================================================
# PART 5: INTERPRETATION AND VALIDATION
# ============================================================================

cat("PART 5: Method Validation\n")
cat("=========================\n\n")

# Compare our bootstrap CI to truth
if(n_successful >= 10) {
  covers_truth <- (true_effects$ate >= ate_ci[1]) & (true_effects$ate <= ate_ci[2])
  ci_width <- ate_ci[2] - ate_ci[1]
  
  cat("BOOTSTRAP VALIDATION:\n")
  cat("- True ATE:", round(true_effects$ate, 4), "\n")
  cat("- Bootstrap estimate:", round(bootstrap_mean, 4), "\n")
  cat("- 95% CI width:", round(ci_width, 4), "\n")
  cat("- CI covers truth:", covers_truth, "\n")
  
  if(covers_truth) {
    cat("✓ Great! Our confidence interval captures the true effect.\n")
  } else {
    cat("⚠ Our CI missed the truth - this can happen ~5% of the time.\n")
  }
  cat("\n")
}

# ============================================================================
# PART 6: VISUALIZATION
# ============================================================================

cat("PART 6: Results Visualization\n")
cat("=============================\n\n")

# Set up plotting
par(mfrow = c(2, 2))

# Plot 1: Bootstrap distribution
if(n_successful >= 10) {
  hist(bootstrap_results$ate, breaks = 15, 
       main = "Bootstrap Distribution of ATE",
       xlab = "Estimated ATE", 
       col = "lightblue",
       freq = FALSE)
  abline(v = true_effects$ate, col = "red", lwd = 2, lty = 1)
  abline(v = gcomp_results$ate, col = "blue", lwd = 2, lty = 2)
  abline(v = ate_ci, col = "green", lwd = 2, lty = 3)
  legend("topright", 
         c("Truth", "Estimate", "95% CI"), 
         col = c("red", "blue", "green"),
         lty = c(1, 2, 3), lwd = 2, cex = 0.7)
}

# Plot 2: Bias assessment
methods_comparison <- data.frame(
  Method = c("Naive", "G-Computation"),
  Estimate = c(diff(treated_outcomes$outcome_rate), gcomp_results$ate),
  Bias = c(diff(treated_outcomes$outcome_rate) - true_effects$ate, 
           gcomp_results$ate - true_effects$ate)
)

barplot(methods_comparison$Bias, 
        names.arg = methods_comparison$Method,
        main = "Bias Comparison",
        ylab = "Bias (Estimate - Truth)",
        col = c("red", "lightblue"),
        ylim = range(methods_comparison$Bias) * 1.2)
abline(h = 0, col = "black", lwd = 2)

# Plot 3: Predicted vs observed outcomes
plot(main_data$true_prob, gcomp_results$pred_control,
     pch = 16, col = alpha("blue", 0.5),
     xlab = "True Probability", ylab = "Predicted Probability",
     main = "Model Calibration\n(Control Group)")
abline(0, 1, col = "red", lwd = 2)
grid()

# Plot 4: Treatment effect by subgroup
main_data$age_group <- cut(main_data$age, breaks = c(0, 60, 70, 100), 
                           labels = c("<60", "60-70", "70+"))
subgroup_effects <- main_data %>% 
  group_by(age_group) %>%
  summarise(
    n = n(),
    observed_ate = mean(outcome[treatment==1]) - mean(outcome[treatment==0]),
    .groups = 'drop'
  ) %>%
  filter(n >= 20)  # Only show groups with sufficient sample size

if(nrow(subgroup_effects) > 0) {
  barplot(subgroup_effects$observed_ate, 
          names.arg = subgroup_effects$age_group,
          main = "Observed ATE by Age Group",
          ylab = "Risk Difference",
          col = "lightcoral")
  abline(h = true_effects$ate, col = "blue", lwd = 2)
  legend("topright", "True ATE", col = "blue", lty = 1, lwd = 2)
}

par(mfrow = c(1, 1))

# ============================================================================
# PART 7: SUMMARY AND DISCUSSION
# ============================================================================

cat("PART 7: Summary and Key Insights\n")
cat("================================\n\n")

# Summary table
summary_table <- data.frame(
  Method = c("Naive (biased)", "G-Computation", "Bootstrap Mean"),
  Estimate = c(
    diff(treated_outcomes$outcome_rate),
    gcomp_results$ate,
    ifelse(exists("bootstrap_mean"), bootstrap_mean, NA)
  ),
  Bias = c(
    diff(treated_outcomes$outcome_rate) - true_effects$ate,
    gcomp_results$ate - true_effects$ate,
    ifelse(exists("bootstrap_mean"), bootstrap_mean - true_effects$ate, NA)
  )
)

cat("SUMMARY COMPARISON:\n")
print(summary_table, 4)
cat("\nTrue ATE:", round(true_effects$ate, 4), "\n\n")

cat("KEY FINDINGS:\n")
cat("1. Confounding bias:", round(abs(diff(treated_outcomes$outcome_rate) - true_effects$ate), 4), "\n")
cat("2. G-computation bias:", round(abs(gcomp_results$ate - true_effects$ate), 4), "\n")
if(exists("covers_truth")) {
  cat("3. CI covers truth:", covers_truth, "\n")
}
cat("4. SuperLearner enables flexible modeling\n")
cat("5. Bootstrap provides valid inference\n\n")

# ============================================================================
# DISCUSSION QUESTIONS
# ============================================================================

cat("=== DISCUSSION QUESTIONS ===\n")
cat("1. Why does the naive analysis give a biased estimate?\n")
cat("2. How well did G-computation correct for confounding?\n")
cat("3. What would happen if we had unmeasured confounders?\n")
cat("4. How does SuperLearner help compared to simple GLM?\n")
cat("5. What assumptions does G-computation rely on?\n")
cat("6. How would you improve the analysis in practice?\n")
cat("7. When might bootstrap CIs be too narrow or too wide?\n\n")

cat("=== KEY TAKEAWAYS ===\n")
cat("- G-computation is a substitution estimator for causal effects\n")
cat("- SuperLearner provides flexible outcome modeling\n")
cat("- Bootstrap gives valid confidence intervals\n")
cat("- Method validation against known truth builds confidence\n")
cat("- Confounding adjustment is crucial for valid causal inference\n\n")

cat("=== LAB COMPLETE ===\n")
cat("Great work implementing G-computation with bootstrap inference!\n")