# ============================================================================
# Manual Cross-Validation Lab: Building Your Own SuperLearner
# Learning Objective: Understand what SuperLearner does "under the hood"
# ============================================================================

# Install/load required packages
if (!require(rpart)) install.packages("rpart")
if (!require(randomForest)) install.packages("randomForest")
library(rpart)
library(randomForest)

# ============================================================================
# PART 1: LOAD AND EXPLORE DATA
# ============================================================================

# Use built-in mtcars dataset - predict mpg from car characteristics
data(mtcars)
head(mtcars)

# Our prediction task: predict mpg (fuel efficiency) from car features
outcome <- "mpg"
predictors <- c("cyl", "disp", "hp", "wt", "qsec", "vs", "am", "gear", "carb")

# Create our analysis dataset
dat <- mtcars[, c(outcome, predictors)]
n <- nrow(dat)

cat("Dataset Info:\n")
cat("- Sample size:", n, "\n")
cat("- Outcome:", outcome, "(mean =", round(mean(dat[[outcome]]), 2), ")\n")
cat("- Number of predictors:", length(predictors), "\n\n")

# ============================================================================
# PART 2: SET UP MANUAL CROSS-VALIDATION
# ============================================================================

# Set random seed for reproducible results
set.seed(123)

# Choose number of folds
V <- 5  # 5-fold cross-validation
cat("Using", V, "-fold cross-validation\n\n")

# Create fold assignments
# Randomly assign each observation to a fold (1, 2, 3, 4, or 5)
fold_id <- sample(rep(1:V, length.out = n))

# Check fold sizes
table(fold_id)

# ============================================================================
# PART 3: DEFINE OUR THREE ALGORITHMS
# ============================================================================

# Algorithm 1: Linear Regression (GLM)
fit_glm <- function(train_data) {
  lm(mpg ~ ., data = train_data)
}

predict_glm <- function(model, test_data) {
  predict(model, newdata = test_data)
}

# Algorithm 2: Decision Tree
fit_tree <- function(train_data) {
  rpart(mpg ~ ., data = train_data, method = "anova")
}

predict_tree <- function(model, test_data) {
  predict(model, newdata = test_data)
}

# Algorithm 3: Random Forest
fit_rf <- function(train_data) {
  randomForest(mpg ~ ., data = train_data, ntree = 100)
}

predict_rf <- function(model, test_data) {
  predict(model, newdata = test_data)
}

# ============================================================================
# PART 4: MANUAL CROSS-VALIDATION LOOP
# ============================================================================

# Initialize storage for results
cv_results <- data.frame(
  fold = 1:V,
  glm_mse = NA,
  tree_mse = NA,
  rf_mse = NA,
  glm_mae = NA,
  tree_mae = NA,
  rf_mae = NA
)

# Storage for all predictions (we'll need these later!)
all_predictions <- data.frame(
  obs_id = 1:n,
  fold = fold_id,
  true_mpg = dat$mpg,
  glm_pred = NA,
  tree_pred = NA,
  rf_pred = NA
)

cat("Running cross-validation...\n")

# THE MAIN CROSS-VALIDATION LOOP
for(v in 1:V) {
  
  cat("Processing fold", v, "of", V, "\n")
  
  # Step 1: Split data into training and validation sets
  train_data <- dat[fold_id != v, ]  # 80% of data
  test_data <- dat[fold_id == v, ]   # 20% of data
  test_indices <- which(fold_id == v)
  
  cat("  - Training size:", nrow(train_data), "\n")
  cat("  - Test size:", nrow(test_data), "\n")
  
  # Step 2: Fit all three algorithms on training data
  glm_model <- fit_glm(train_data)
  tree_model <- fit_tree(train_data)
  rf_model <- fit_rf(train_data)
  
  # Step 3: Make predictions on test data
  glm_preds <- predict_glm(glm_model, test_data)
  tree_preds <- predict_tree(tree_model, test_data)
  rf_preds <- predict_rf(rf_model, test_data)
  
  # Step 4: Calculate errors for this fold
  true_values <- test_data$mpg
  
  # Mean Squared Error (MSE)
  glm_mse <- mean((true_values - glm_preds)^2)
  tree_mse <- mean((true_values - tree_preds)^2)
  rf_mse <- mean((true_values - rf_preds)^2)
  
  # Mean Absolute Error (MAE) 
  glm_mae <- mean(abs(true_values - glm_preds))
  tree_mae <- mean(abs(true_values - tree_preds))
  rf_mae <- mean(abs(true_values - rf_preds))
  
  # Step 5: Store results
  cv_results[v, ] <- c(v, glm_mse, tree_mse, rf_mse, glm_mae, tree_mae, rf_mae)
  
  # Store predictions for later analysis
  all_predictions[test_indices, "glm_pred"] <- glm_preds
  all_predictions[test_indices, "tree_pred"] <- tree_preds
  all_predictions[test_indices, "rf_pred"] <- rf_preds
  
  cat("  - GLM MSE:", round(glm_mse, 3), "| Tree MSE:", round(tree_mse, 3), 
      "| RF MSE:", round(rf_mse, 3), "\n")
}

cat("\nCross-validation complete!\n\n")

# ============================================================================
# PART 5: SUMMARIZE RESULTS
# ============================================================================

cat("=== CROSS-VALIDATION RESULTS ===\n\n")

# Print fold-by-fold results
print(cv_results)

# Calculate average performance across folds
avg_results <- data.frame(
  Algorithm = c("Linear Regression", "Decision Tree", "Random Forest"),
  Avg_MSE = c(mean(cv_results$glm_mse), mean(cv_results$tree_mse), mean(cv_results$rf_mse)),
  Avg_MAE = c(mean(cv_results$glm_mae), mean(cv_results$tree_mae), mean(cv_results$rf_mae)),
  SD_MSE = c(sd(cv_results$glm_mse), sd(cv_results$tree_mse), sd(cv_results$rf_mse))
)

cat("\n=== AVERAGE PERFORMANCE ===\n")
print(avg_results)

# Determine winner
best_mse <- which.min(avg_results$Avg_MSE)
best_mae <- which.min(avg_results$Avg_MAE)

cat("\n=== WINNER ===\n")
cat("Best MSE:", avg_results$Algorithm[best_mse], 
    "(", round(avg_results$Avg_MSE[best_mse], 3), ")\n")
cat("Best MAE:", avg_results$Algorithm[best_mae], 
    "(", round(avg_results$Avg_MAE[best_mae], 3), ")\n")

# ============================================================================
# PART 6: CREATE A SIMPLE "SUPERLEARNER"
# ============================================================================

cat("\n=== BUILDING SIMPLE SUPERLEARNER ===\n")

# Method 1: Simple average (equal weights)
all_predictions$simple_avg <- (all_predictions$glm_pred + all_predictions$tree_pred + all_predictions$rf_pred) / 3

# Method 2: Weighted average based on CV performance
# Give more weight to algorithms with lower MSE
glm_weight <- 1 / avg_results$Avg_MSE[1]
tree_weight <- 1 / avg_results$Avg_MSE[2]
rf_weight <- 1 / avg_results$Avg_MSE[3]
total_weight <- glm_weight + tree_weight + rf_weight

# Normalize weights to sum to 1
glm_weight <- glm_weight / total_weight
tree_weight <- tree_weight / total_weight
rf_weight <- rf_weight / total_weight

cat("Optimal weights:\n")
cat("- GLM weight:", round(glm_weight, 3), "\n")
cat("- Tree weight:", round(tree_weight, 3), "\n")
cat("- RF weight:", round(rf_weight, 3), "\n")

# Create weighted combination
all_predictions$weighted_avg <- glm_weight * all_predictions$glm_pred + 
  tree_weight * all_predictions$tree_pred +
  rf_weight * all_predictions$rf_pred

# ============================================================================
# PART 7: EVALUATE OUR "SUPERLEARNER"
# ============================================================================

# Calculate MSE for each approach
individual_performance <- data.frame(
  Method = c("GLM Only", "Tree Only", "RF Only", "Simple Average", "Weighted Average"),
  MSE = c(
    mean((all_predictions$true_mpg - all_predictions$glm_pred)^2),
    mean((all_predictions$true_mpg - all_predictions$tree_pred)^2),
    mean((all_predictions$true_mpg - all_predictions$rf_pred)^2),
    mean((all_predictions$true_mpg - all_predictions$simple_avg)^2),
    mean((all_predictions$true_mpg - all_predictions$weighted_avg)^2)
  )
)

cat("\n=== FINAL PERFORMANCE COMPARISON ===\n")
print(individual_performance)

# Find the best approach
best_overall <- which.min(individual_performance$MSE)
cat("\nOverall winner:", individual_performance$Method[best_overall], "\n")
cat("Best MSE:", round(individual_performance$MSE[best_overall], 3), "\n")

# ============================================================================
# PART 8: VISUALIZE RESULTS
# ============================================================================

# Create scatter plot of predictions vs true values
par(mfrow = c(2, 3))  # Changed to 2x3 grid to accommodate 5 plots

# GLM predictions
plot(all_predictions$true_mpg, all_predictions$glm_pred,
     main = "GLM Predictions", xlab = "True MPG", ylab = "Predicted MPG",
     pch = 16, col = "blue")
abline(0, 1, col = "red", lty = 2)  # Perfect prediction line

# Tree predictions  
plot(all_predictions$true_mpg, all_predictions$tree_pred,
     main = "Tree Predictions", xlab = "True MPG", ylab = "Predicted MPG", 
     pch = 16, col = "green")
abline(0, 1, col = "red", lty = 2)

# Random Forest predictions
plot(all_predictions$true_mpg, all_predictions$rf_pred,
     main = "Random Forest Predictions", xlab = "True MPG", ylab = "Predicted MPG",
     pch = 16, col = "darkgreen")
abline(0, 1, col = "red", lty = 2)

# Simple average
plot(all_predictions$true_mpg, all_predictions$simple_avg,
     main = "Simple Average", xlab = "True MPG", ylab = "Predicted MPG",
     pch = 16, col = "purple")
abline(0, 1, col = "red", lty = 2)

# Weighted average  
plot(all_predictions$true_mpg, all_predictions$weighted_avg,
     main = "Weighted Average", xlab = "True MPG", ylab = "Predicted MPG",
     pch = 16, col = "orange")
abline(0, 1, col = "red", lty = 2)

par(mfrow = c(1, 1))  # Reset plotting

# ============================================================================
# PART 9: DISCUSSION QUESTIONS
# ============================================================================

cat("\n=== DISCUSSION QUESTIONS ===\n")
cat("1. Which algorithm performed better individually? Why do you think this is?\n")
cat("2. Did the ensemble (average) beat all individual algorithms?\n") 
cat("3. How much did the weighted average improve over simple average?\n")
cat("4. Looking at the plots, which method has predictions closest to the red line?\n")
cat("5. How did Random Forest compare to the simpler methods?\n")
cat("6. What would happen if we added more algorithms to our library?\n")
cat("7. How might results change with a different dataset or more folds?\n\n")

# ============================================================================
# BONUS: COMPARE TO REAL SUPERLEARNER (IF TIME PERMITS)
# ============================================================================

# Uncomment and run this section if SuperLearner package is available
# if (require(SuperLearner)) {
#   cat("=== COMPARISON TO REAL SUPERLEARNER ===\n")
#   
#   # Prepare data for SuperLearner
#   Y <- dat$mpg
#   X <- dat[, predictors]
#   
#   # Define library (same algorithms we used)
#   SL.library <- c("SL.lm", "SL.rpart", "SL.randomForest")
#   
#   # Fit SuperLearner
#   sl_fit <- SuperLearner(Y = Y, X = X, SL.library = SL.library, 
#                          cvControl = list(V = 5))
#   
#   cat("SuperLearner weights:\n")
#   print(sl_fit$coef)
#   
#   cat("SuperLearner CV Risk:", sl_fit$cvRisk, "\n")
#   cat("Our manual CV MSEs:", avg_results$Avg_MSE, "\n")
# }

cat("Lab complete! Great job building your own SuperLearner with Random Forest!\n")