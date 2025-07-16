# ============================================================================ 
#  IPTW Lab – stochastic-intervention version (+20-pp propensity-score shift)
#  SSE 708: Machine Learning in the Era of Big Data
#  Goal:  estimate  ψ = E[Y]  under  g*(A=1|W) = min{ p̂(A=1|W) + 0.20 , 0.99 }
# ============================================================================

## ---------- 0.  package setup ------------------------------------------------
if (!require(SuperLearner)) install.packages("SuperLearner")
if (!require(randomForest)) install.packages("randomForest")
if (!require(ggplot2))      install.packages("ggplot2")
if (!require(dplyr))        install.packages("dplyr")

library(SuperLearner); library(randomForest)
library(ggplot2);      library(dplyr)

cat("=== IPTW + Influence-Function CI for a 20-pp treatment-probability shift ===\n\n")

## ---------- 1.  simulate HeartFix data --------------------------------------
generate_data <- function(n = 1000, seed = NULL){
  if(!is.null(seed)) set.seed(seed)
  age <- pmin( pmax(round(rnorm(n,68,12)), 30), 90)
  nyha<- sample(1:4,n,TRUE, prob=c(.1,.4,.4,.1))
  ef  <- pmin( pmax(round(rnorm(n,35,10)),15),60)
  bnp <- exp(rnorm(n, 6 + 0.15*nyha - 0.02*ef + 0.01*age ,0.4))
  diabetes<- rbinom(n,1, plogis(-1 + 0.02*age + 0.3*nyha))
  ckd     <- rbinom(n,1, plogis(-2 + 0.03*age + 0.2*nyha))
  
  logit_ps<- -1.5 + 0.02*age + 0.4*nyha - 0.01*ef +
    0.2*log(bnp/400)+0.3*diabetes+0.4*ckd
  ps  <- plogis(logit_ps)
  A   <- rbinom(n,1,ps)
  
  logit_y <- -2.5 + 0.03*age + 0.6*nyha - 0.015*ef +
    0.3*log(bnp/400)+0.4*diabetes +0.3*ckd - 1.0*A
  Y  <- rbinom(n,1,plogis(logit_y))
  
  data.frame(id=1:n, age, nyha, ef, bnp, 
             diabetes, ckd, 
             treatment=A, outcome=Y,
             true_ps=ps)
}

set.seed(12345)
main_data <- generate_data(1000)

## ---------- 2.  IPTW for a +20-pp propensity shift --------------------------
iptw_shift <- function(dat, shift = 0.20){
  X <- dat[,c("age","nyha","ef","bnp","diabetes","ckd")]
  A <- dat$treatment;  Y <- dat$outcome
  
  sl_lib <- c("SL.glm","SL.randomForest","SL.mean")
  ps_fit <- SuperLearner(Y = A, X = X,
                         SL.library = sl_lib, family = binomial())
  p_hat  <- pmin(pmax(predict(ps_fit)$pred, 0.001), 0.999)
  
  ## target policy g*(A=1|W) : add 20 pp, cap at 0.99
  p_star <- pmin(p_hat + shift, 0.99)
  
  ## weights  w_i = g*(A|W) / ĝ(A|W)
  w_raw <- ifelse(A==1,  p_star / p_hat,
                  (1-p_star)/(1-p_hat))
  w     <- w_raw / mean(w_raw)          # Hájek normalisation
  
  ## plug-in estimator of counterfactual mean under policy shift
  psi_hat <- sum(w * Y) / sum(w)        # ≡ mean(Y * w_normalised)
  
  list(psi = psi_hat, w = w,
       p_hat = p_hat, p_star = p_star,
       model = ps_fit)
}

shift_res <- iptw_shift(main_data, shift = 0.20)
cat("Estimated hospitalisation risk if uptake ↑ by 20 pp :",
    round(shift_res$psi,4), "\n\n")

## ---------- 3.  ORACLE risk under same shift --------------------------------
truth_shift <- with(main_data,{
  ## true outcome probabilities if A=1 and A=0
  p1 <- plogis(-2.5 +0.03*age +0.6*nyha -0.015*ef + 
                 0.3*log(bnp/400)+0.4*diabetes +0.3*ckd -1.0)
  p0 <- plogis(-2.5 +0.03*age +0.6*nyha -0.015*ef + 
                 0.3*log(bnp/400)+0.4*diabetes +0.3*ckd)
  
  p_star <- pmin(shift_res$p_hat + 0.20, 0.99)
  mean( p_star * p1 + (1-p_star)*p0 )
})

cat("Oracle risk under +20 pp policy              :",
    round(truth_shift,4), "\n\n")

## ---------- 4.  Influence-function SE & Wald CI -----------------------------
psi_hat <- shift_res$psi
w       <- shift_res$w
Y       <- main_data$outcome
n       <- length(Y)

IF  <- (w) * (Y - psi_hat)              # IF for Hájek-normalised mean
se_if <- sqrt( mean(IF^2) / n )
ci    <- psi_hat + c(-1,1)*qnorm(.975)*se_if

cat("Influence-function SE :", round(se_if,4), "\n")
cat("95% Wald CI           :", round(ci[1],4),"–",round(ci[2],4), "\n\n")

## ---------- 5.  Truth vs estimate table & plots -----------------------------
natural_risk <- mean(Y)
tab <- data.frame(
  Scenario = c("Natural course",
               "20 pp shift  –  truth",
               "20 pp shift  –  estimate"),
  Risk     = c(natural_risk, truth_shift, psi_hat),
  SE       = c(NA, NA, se_if),
  LCI      = c(NA, NA, ci[1]),
  UCI      = c(NA, NA, ci[2]))

print(tab,4); cat("\n")

par(mfrow=c(1,2))

## (1) barplot truth vs estimate
bp_col <- c("grey70","darkgreen","dodgerblue")
barplot(tab$Risk,
        names.arg = tab$Scenario,
        col = bp_col, ylim=c(0,max(tab$Risk)*1.2),
        ylab="6-month hospitalisation risk",
        main="Counterfactual risk\ntruth vs estimate")
arrows(x0=3, x1=3,
       y0=ci[1], y1=ci[2],
       angle=90, code=3, length=0.07, lwd=2)

## (2) Propensity score shift - FIXED VERSION
# Calculate densities first to get proper xlim
dens_orig <- density(shift_res$p_hat)
dens_shift <- density(shift_res$p_star)

# Get the full range needed for both densities
xlim_range <- range(c(dens_orig$x, dens_shift$x))

# Plot with extended x-axis
plot(dens_orig, col="black", lwd=2,
     main="Propensity scores", xlab="p(A=1 | W)",
     xlim = xlim_range)
lines(dens_shift, col="darkgreen", lwd=2, lty=2)
legend("topright", c("Observed p̂", "Shifted p*"),
       col=c("black","darkgreen"), lty=c(1,2), lwd=2, bty="n")

par(mfrow=c(1,1))

## ---------- 6.  Discussion prompts -----------------------------------------
cat("=== DISCUSSION ===\n")
cat("• Natural-course risk  =", round(natural_risk,4), "\n")
cat("• Oracle 20 pp-shift   =", round(truth_shift ,4), "\n")
cat("• Estimate (+95 % CI)  =", round(psi_hat     ,4),
    paste0(" (", round(ci[1],4),"–",round(ci[2],4),")\n"))
cat("Does the CI cover the truth? ",
    truth_shift >= ci[1] & truth_shift <= ci[2], "\n")

cat("\nTry:  shift_res <- iptw_shift(main_data, shift = -0.20)  # ↓20 pp uptake\n")
cat("      inspect weights & bias as the shift magnitude grows.\n")
cat("=== END OF LAB ===\n")