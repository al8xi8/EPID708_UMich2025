# ------------------------------------------------------------------
#  TMLE vs. Wald · 2 : 1 randomised device-trial with mild imbalance
# ------------------------------------------------------------------
library(tmle)
library(SuperLearner)
library(dplyr)      ; library(tidyr)
library(ggplot2)    ; library(gridExtra)

set.seed(708)

# --------------------------- 1. Data generator --------------------
simulate_device_trial <- function(n = 350,
                                  treat_ratio = 2/3,        # ≈2 : 1
                                  imbalance   = 0.3) {      # 0 = perfect randomisation
  age  <- rnorm(n, 70, 10)
  male <- rbinom(n, 1, .60)
  nyha <- sample(2:4, n, TRUE, c(.30, .50, .20))
  bnp  <- rlnorm(n, log(300), .8)
  ef   <- rnorm(n, 35, 10)
  
  age_s <- (age-70)/10
  bnp_s <- (log(bnp)-log(300))/.8
  ef_s  <- (ef-35)/10
  
  ## ---- treatment assignment: 2 : 1 + modest imbalance -----------
  base_logit <- qlogis(treat_ratio)                 # sets overall 2 : 1
  linpred     <- base_logit +
    imbalance*(0.4*bnp_s + 0.6*(nyha-3))
  p_treat     <- plogis(linpred)
  A           <- rbinom(n, 1, p_treat)
  
  ## ---- outcome model (HF hospitalisations) -----------------------
  lambda0 <- exp(-0.5 + 0.3*age_s + 0.2*male + 0.4*(nyha-3) +
                   0.3*bnp_s - 0.2*ef_s)
  tau     <- -0.8 + 0.3*(nyha-3) + 0.2*bnp_s - 0.1*age_s   # effect-modification
  Y       <- rpois(n, lambda0 * exp(A*tau))
  
  data.frame(age, male, nyha, bnp, ef, A, Y)
}

# ------------------------- 2. Single-trial analysis ---------------
analyse_one <- function(dat) {
  
  ## --- Wald Poisson rate difference ------------------------------
  n0 <- sum(dat$A==0);  y0 <- sum(dat$Y[dat$A==0])
  n1 <- sum(dat$A==1);  y1 <- sum(dat$Y[dat$A==1])
  r0 <- y0/n0;          r1 <- y1/n1
  est_w <- r1 - r0
  se_w  <- sqrt(r0/n0 + r1/n1)
  z97   <- qnorm(.975)
  ci_w  <- est_w + c(-1,1)*z97*se_w
  
  ## --- TMLE with SL (propensity + outcome) -----------------------
  Wmat    <- dat[, c("age","male","nyha","bnp","ef")]
  SL.lib  <- c("SL.mean","SL.glm","SL.glm.interaction","SL.ranger")
  
  tm      <- tmle(Y = dat$Y,
                  A = dat$A,
                  W = Wmat,
                  family       = "poisson",
                  Q.SL.library = SL.lib,      # outcome model
                  g.SL.library = "SL.glm")    # learns p(A=1|W)
  
  est_t <- tm$estimates$ATE$psi
  se_t  <- sqrt(tm$estimates$ATE$var.psi)
  ci_t  <- est_t + c(-1,1)*z97*se_t
  
  tibble(method = factor(c("Wald","TMLE"), levels = c("Wald","TMLE")),
         est    = c(est_w, est_t),
         se     = c(se_w , se_t ),
         lwr    = c(ci_w[1], ci_t[1]),
         upr    = c(ci_w[2], ci_t[2]))
}

# ---------------------- 3. “True” marginal ATE --------------------
true_ate <- (function(M = 2e5){
  d  <- simulate_device_trial(M)
  d$A <- 0
  age_s <- (d$age-70)/10
  bnp_s <- (log(d$bnp)-log(300))/.8
  ef_s  <- (d$ef-35)/10
  lambda0 <- exp(-0.5 + 0.3*age_s + 0.2*d$male + 0.4*(d$nyha-3) +
                   0.3*bnp_s - 0.2*ef_s)
  tau <- -0.8 + 0.3*(d$nyha-3) + 0.2*bnp_s - 0.1*age_s
  mean(lambda0*exp(tau)) - mean(lambda0)
})()

# -------------------- 4. Monte-Carlo simulation -------------------
nsim <- 200                  # set smaller (e.g. 50) when testing
message("Running ", nsim, " simulations …")
sim_res <- replicate(nsim, analyse_one(simulate_device_trial()), simplify = FALSE) |>
  bind_rows(.id = "sim") |>
  mutate(sim = as.integer(sim))

# ---------------------- 5. Summary diagnostics --------------------
summary_tbl <- sim_res |>
  group_by(method) |>
  summarise(mean_est = mean(est),
            bias     = mean(est - true_ate),
            sd_est   = sd(est),
            mean_se  = mean(se),
            coverage = mean(lwr <= true_ate & upr >= true_ate),
            .groups  = "drop")

print(summary_tbl, digits = 3)

# efficiency gain (SE)
eff <- sim_res |>
  pivot_wider(id_cols = sim, names_from = method, values_from = se) |>
  mutate(gain = 100*(Wald - TMLE)/Wald)
cat(sprintf("\nAverage efficiency gain (SE): %.1f %%\n",
            mean(eff$gain, na.rm = TRUE)))

# ----------------------- 6. Figures for slides --------------------
bins <- 30

## 6·1  Sampling distribution
p_hist <- ggplot(sim_res, aes(est, fill = method)) +
  geom_histogram(alpha = .6, bins = bins, position = "identity") +
  geom_vline(xintercept = true_ate, linetype = "dashed") +
  facet_wrap(~method, ncol = 1) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none") +
  labs(title = "Sampling distribution",
       subtitle = paste("True ATE =", round(true_ate, 2)),
       x = "ATE estimate", y = "Count")

## 6·2  95 % CI width
p_width <- sim_res |> mutate(width = upr - lwr) |>
  ggplot(aes(method, width, fill = method)) +
  geom_boxplot(outlier.alpha = .3) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none") +
  labs(title = "95 % CI width", x = NULL, y = "Width")

## 6·3  Cumulative coverage
cov_df <- sim_res |>
  mutate(covered = lwr <= true_ate & upr >= true_ate) |>
  group_by(method, sim) |>
  summarise(covered = first(covered), .groups = "drop") |>
  group_by(method) |>
  mutate(cum_cov = cumsum(covered) / row_number())

p_cov <- ggplot(cov_df, aes(sim, cum_cov, colour = method)) +
  geom_line(size = 1) +
  geom_hline(yintercept = 0.95, linetype = "dashed") +
  scale_y_continuous(limits = c(0.75, 1)) +
  theme_minimal(base_size = 12) +
  labs(title = "Cumulative 95 % coverage",
       x = "Simulation", y = "Coverage")

## 6·4  Arrange & save
fig_grid <- grid.arrange(p_hist, p_width, p_cov, ncol = 3)
ggsave("tmle_vs_wald_device_trial.png", fig_grid,
       width = 12, height = 4, dpi = 300)

# ---------------------- 7. Save raw results -----------------------
tryCatch(
  {
    write.csv(sim_res, "tmle_simulation_results.csv", row.names = FALSE)
    message("Raw results saved to tmle_simulation_results.csv")
  },
  error = function(e) message("Unable to write CSV (read-only FS?).")
)
