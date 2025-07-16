# Practice Problem 3: Simulate Causal Data and Calculate True ATE
# For the trauma/platelet example: W -> A -> Y

# Function to simulate data from our causal model
simulate_causal_data <- function(n = 1000) {
  
  # Step 1: Generate baseline covariates W
  # Let's say W represents patient severity (0 = less severe, 1 = more severe)
  W <- rbinom(n, size = 1, prob = 0.4)  # 40% have severe injuries
  
  # Step 2: Generate treatment A based on W (confounding!)
  # Sicker patients more likely to get platelets
  prob_treatment <- 0.3 + 0.4 * W  # P(A=1) = 0.3 if W=0, 0.7 if W=1
  A <- rbinom(n, size = 1, prob = prob_treatment)
  
  # Step 3: Generate outcome Y based on BOTH W and A
  # Sicker patients (W=1) have higher mortality
  # Platelets (A=1) reduce mortality
  prob_death <- 0.2 + 0.3 * W - 0.15 * A  # Base rate + severity effect - treatment effect
  Y <- rbinom(n, size = 1, prob = prob_death)
  
  # Return the observed data
  data.frame(W = W, A = A, Y = Y)
}

# Function to calculate TRUE ATE by generating counterfactuals
calculate_true_ate <- function(n = 10000) {
  
  # Generate baseline covariates (same as above)
  W <- rbinom(n, size = 1, prob = 0.4)
  
  # For each person, calculate BOTH potential outcomes
  # Y_0: What would happen if they got NO treatment (A = 0)
  prob_death_no_treatment <- 0.2 + 0.3 * W - 0.15 * 0  # A = 0
  Y_0 <- rbinom(n, size = 1, prob = prob_death_no_treatment)
  
  # Y_1: What would happen if they got treatment (A = 1) 
  prob_death_with_treatment <- 0.2 + 0.3 * W - 0.15 * 1  # A = 1
  Y_1 <- rbinom(n, size = 1, prob = prob_death_with_treatment)
  
  # Calculate the Average Treatment Effect
  # ATE = E[Y_1 - Y_0] = Average difference in outcomes
  individual_effects <- Y_1 - Y_0  # Individual causal effects
  true_ate <- mean(individual_effects)
  
  # Also calculate the marginal probabilities for interpretation
  prob_death_if_all_treated <- mean(Y_1)
  prob_death_if_none_treated <- mean(Y_0)
  
  # Return results
  list(
    true_ate = true_ate,
    prob_death_if_all_treated = prob_death_if_all_treated,
    prob_death_if_none_treated = prob_death_if_none_treated,
    interpretation = paste("If everyone got platelets, death rate would be", 
                           round(prob_death_if_all_treated, 3),
                           "vs", round(prob_death_if_none_treated, 3), 
                           "if no one got platelets")
  )
}

# Function to demonstrate confounding in observed data
analyze_observed_data <- function(data) {
  
  # Crude (confounded) analysis - just compare treated vs untreated
  death_rate_treated <- mean(data$Y[data$A == 1])
  death_rate_untreated <- mean(data$Y[data$A == 0])
  crude_difference <- death_rate_treated - death_rate_untreated
  
  # Stratified analysis (controlling for W)
  # Among less severe patients (W = 0)
  less_severe <- data[data$W == 0, ]
  death_rate_treated_mild <- mean(less_severe$Y[less_severe$A == 1])
  death_rate_untreated_mild <- mean(less_severe$Y[less_severe$A == 0])
  
  # Among more severe patients (W = 1)  
  more_severe <- data[data$W == 1, ]
  death_rate_treated_severe <- mean(more_severe$Y[more_severe$A == 1])
  death_rate_untreated_severe <- mean(more_severe$Y[more_severe$A == 0])
  
  # Weighted average of stratified effects (closer to true ATE)
  prop_mild <- mean(data$W == 0)
  prop_severe <- mean(data$W == 1)
  stratified_ate <- prop_mild * (death_rate_treated_mild - death_rate_untreated_mild) + 
    prop_severe * (death_rate_treated_severe - death_rate_untreated_severe)
  
  # Return comparison
  list(
    crude_difference = crude_difference,
    stratified_ate = stratified_ate,
    breakdown = data.frame(
      Group = c("Less Severe (W=0)", "More Severe (W=1)"),
      Treated_Death_Rate = c(death_rate_treated_mild, death_rate_treated_severe),
      Untreated_Death_Rate = c(death_rate_untreated_mild, death_rate_untreated_severe),
      Difference = c(death_rate_treated_mild - death_rate_untreated_mild,
                     death_rate_treated_severe - death_rate_untreated_severe)
    )
  )
}

# ============================================================================
# RUN THE SIMULATION
# ============================================================================

# Set seed for reproducible results
set.seed(123)

# 1. Calculate the TRUE ATE using counterfactuals
cat("=== TRUE CAUSAL EFFECT ===\n")
true_results <- calculate_true_ate(n = 50000)  # Large sample for precision
cat("True ATE:", round(true_results$true_ate, 4), "\n")
cat(true_results$interpretation, "\n\n")

# 2. Generate observed data (what we actually see in practice)
cat("=== OBSERVED DATA ANALYSIS ===\n")
observed_data <- simulate_causal_data(n = 5000)

# 3. Analyze observed data to see confounding
observed_results <- analyze_observed_data(observed_data)

cat("Crude analysis (BIASED due to confounding):\n")
cat("  Death rate difference (Treated - Untreated):", round(observed_results$crude_difference, 4), "\n\n")

cat("Stratified analysis (UNBIASED - controls for confounding):\n")
print(observed_results$breakdown)
cat("\nWeighted average of stratified effects:", round(observed_results$stratified_ate, 4), "\n")

# 4. Compare to true ATE
cat("\n=== COMPARISON ===\n")
cat("True ATE:                  ", round(true_results$true_ate, 4), "\n")
cat("Crude analysis:            ", round(observed_results$crude_difference, 4), "\n") 
cat("Stratified analysis:       ", round(observed_results$stratified_ate, 4), "\n")
cat("\nThe stratified analysis should be close to the true ATE!\n")
cat("The crude analysis is biased because sicker patients get platelets more often.\n")