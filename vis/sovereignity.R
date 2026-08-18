library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)

suppressMessages(library(tidyverse))
folder <- dirname(rstudioapi::getSourceEditorContext()$path)

constructs <- read.csv(file.path(folder, '..', 'results', 'final',
                      'federated', 'constructs.csv'))

mmd <- read.csv(file.path(folder, '..', 'results', 'final',
                                 'federated', 'heterogeneity_mmd.csv'))

federated_results <- read.csv(file.path(folder, '..', 'results', 'final',
                          'federated', 'federated_results.csv'))

shapely_df <- read.csv(file.path(folder, '..', 'results', 'final',
                                 'federated', 'shapley.csv'))

federated_results$client = factor(
  federated_results$client,
  levels = c('uganda', 'zimbabwe', 'zambia'),
  labels = c('Uganda', 'Zimbabwe', 'Zambia')
)

summary_df <- constructs %>%
  group_by(client) %>%
  summarise(
    delta_sov_mean = mean(delta_sov_fedavg),
    delta_sov_sd   = sd(delta_sov_fedavg),
    gamma_mean     = mean(gamma_fedavg),
    gamma_sd       = sd(gamma_fedavg),
    .groups = "drop"
  )


# Panel A: MMD vs. Sovereignty Cost ───────────────────────────────────────
avg_mmd <- sapply(c("uganda", "zambia", "zimbabwe"), function(cl) {
  mean(mmd$mmd[grepl(cl, mmd$pair)])
})

panel_a_df <- summary_df %>%
  mutate(mmd = avg_mmd[client])

mmd_plot <- ggplot(panel_a_df, aes(x = mmd, y = delta_sov_mean, color = client)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray") +
  geom_errorbar(aes(ymin = delta_sov_mean - delta_sov_sd,
                    ymax = delta_sov_mean + delta_sov_sd), width = 0.003) +
  geom_point(size = 6) +
  geom_text(aes(label = tools::toTitleCase(client)), hjust = -0.2, vjust = -0.5, size = 4) +
  scale_color_viridis_d(option = "viridis") +
  labs(title = "(A) Data Heterogeneity",
       subtitle = "Countries whose data looks least like their\nneighbors' pay the highest price for not sharing.",
       x = "Pairwise Maximum Mean Discrepancy (MMD)",
       y = expression("Sovereignty Cost " (Delta[sov]))) +
  theme_minimal() + scale_x_continuous(limits = c(0, 0.13)) +
  scale_color_viridis_d(option = "viridis", na.value = "grey90") +
  theme(legend.position = 'none',
        plot.title = element_text(size = 10, face = "bold"),
        plot.subtitle = element_text(size = 8),
        axis.title.y = element_text(size = 8),
        axis.title.x = element_text(size = 8),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(size = 7),
        axis.text.y = element_text(size = 7),
        axis.line.x  = element_line(size = 0.15),
        axis.line.y  = element_line(size = 0.15))

# Panel B: Sovereignty Cost bar chart ─────────────────────────────────────
panel_b_df <- summary_df %>%
  mutate(client = factor(client, levels = c("uganda", "zimbabwe", "zambia")))

sovereignty <- ggplot(panel_b_df, aes(x = client, y = delta_sov_mean, fill = client)) +
  geom_col(width = 0.7) +
  geom_errorbar(aes(ymin = delta_sov_mean - delta_sov_sd,
                    ymax = delta_sov_mean + delta_sov_sd), width = 0.2) +
  geom_hline(yintercept = 0) +
  geom_text(aes(y = ifelse(delta_sov_mean >= 0, delta_sov_mean + delta_sov_sd, 
                           delta_sov_mean - delta_sov_sd), label = round(delta_sov_mean, 3),
                vjust = ifelse(delta_sov_mean >= 0, -0.5, 1.3)),
            size = 2.5) +
  scale_fill_viridis_d(option = "viridis") +
  scale_x_discrete(labels = tools::toTitleCase) +
  labs(title = "(B) Sovereignty Cost",
       subtitle = "The cost of not sharing healthcare training\ndata with partners",
       x = NULL,
       y = expression("Sovereignty Cost " (Delta[sov]))) +
  theme_minimal() + scale_y_continuous(limits = c(-0.05, 0.25)) +
  theme(legend.position = 'none',
        plot.title = element_text(size = 10, face = "bold"),
        plot.subtitle = element_text(size = 8),
        axis.title.y = element_text(size = 8),
        axis.title.x = element_text(size = 8),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(size = 7),
        axis.text.y = element_text(size = 7),
        axis.line.x  = element_line(size = 0.15),
        axis.line.y  = element_line(size = 0.15))


# Panel C: Federation Gain bar chart ──────────────────────────────────────
panel_c_df <- summary_df %>%
  mutate(client = factor(client, levels = c("uganda", "zimbabwe", "zambia")))

federation_gain <- ggplot(panel_c_df, aes(x = client, y = gamma_mean, 
                          fill = client)) + geom_col(width = 0.7) +
  geom_errorbar(aes(ymin = gamma_mean - gamma_sd,
                    ymax = gamma_mean + gamma_sd), width = 0.2) +
  geom_hline(yintercept = 0) +
  geom_text(aes(y = ifelse(gamma_mean >= 0, gamma_mean + gamma_sd, gamma_mean - gamma_sd),
                label = round(gamma_mean, 3),
                vjust = ifelse(gamma_mean >= 0, -0.5, 1.3)),
            size = 2.5) +
  scale_fill_viridis_d(option = "viridis") +
  scale_y_continuous(limits = c(-0.53, 0.06)) +
  scale_x_discrete(labels = tools::toTitleCase) +
  labs(title = "(C) Federation Gain",
       subtitle = "The cost of sharing training data with partner\ncountries.",
       x = NULL,
       y = expression("Federation Gain " (Gamma))) +
  theme_minimal() +
  theme(legend.position = 'none',
        plot.title = element_text(size = 10, face = "bold"),
        plot.subtitle = element_text(size = 8),
        axis.title.y = element_text(size = 8),
        axis.title.x = element_text(size = 8),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(size = 7),
        axis.text.y = element_text(size = 7),
        axis.line.x  = element_line(size = 0.15),
        axis.line.y  = element_line(size = 0.15))


regime_order <- c("climatology", "local", "transfer", "fedavg", "centralised")

spectrum_df <- federated_results %>%
  filter(regime %in% regime_order) %>%
  group_by(regime, client) %>%
  summarise(mean_r2 = mean(R2), sd_r2 = sd(R2), .groups = "drop") %>%
  mutate(regime = factor(regime, levels = regime_order))

spectrum_df$regime = factor(
  spectrum_df$regime,
  levels = c("climatology", "local", "transfer", "fedavg", "centralised"),
  labels = c('Climatology \n(baseline)', 'Local \n(No Sharing)', 
             'Transfer \nLearning', 'Federated', 'Centralised \n(pool data)')
)

client_levels <- c("Uganda", "Zimbabwe", "Zambia")

non_transfer <- federated_results %>%
  filter(regime != "transfer") %>%
  mutate(regime_label = if_else(regime == "fedavg",
                                paste0("fedavg_", weighting),
                                regime)) %>%
  select(regime_label, client, seed, R2)

transfer_source <- federated_results %>%
  filter(regime == "transfer", mode == "source") %>%
  transmute(regime_label = "transfer", client, seed, R2)

transfer_finetuned <- federated_results %>%
  filter(regime == "transfer", mode == "finetuned") %>%
  group_by(client, seed) %>%
  summarise(R2 = mean(R2), .groups = "drop") %>%
  mutate(regime_label = "transfer") %>%
  select(regime_label, client, seed, R2)

plot_df <- bind_rows(non_transfer, transfer_source, transfer_finetuned)


# ── Aggregate: mean ± SD across seeds, per (regime_label, client) ──────────
agg <- plot_df %>%
  group_by(regime_label, client) %>%
  summarise(mean_r2 = mean(R2), sd_r2 = sd(R2), .groups = "drop") %>%
  mutate(sd_r2 = if_else(is.na(sd_r2), 0, sd_r2))

regime_order  <- c("climatology", "local", "transfer",
                   "fedavg_uniform", "fedavg_sample", "centralised")
regime_labels <- c("Climatology\n(baseline)", "Local\n(No Sharing)",
                   "Transfer\n(avg. across sources)", "FedAvg\n(uniform)",
                   "FedAvg\n(sample-wt)", "Centralised\n(pool data)")

agg <- agg %>%
  mutate(regime_label = factor(regime_label, levels = regime_order,
                               labels = regime_labels),
         client = factor(client, levels = client_levels))

# ── Plot ─────────────────────────────────────────────────────────────────
model_regimes <- ggplot(agg, aes(x = regime_label, y = mean_r2, fill = client)) +
  geom_col(position = position_dodge(0.75), width = 0.7) +
  geom_errorbar(aes(ymin = mean_r2 - sd_r2, ymax = mean_r2 + sd_r2),
                position = position_dodge(0.75), width = 0.2) +
  geom_hline(yintercept = 0) +
  geom_text(aes(y = if_else(mean_r2 >= 0, mean_r2 + sd_r2, mean_r2 - sd_r2),
                label = sprintf("%.2f", mean_r2),
                vjust = if_else(mean_r2 >= 0, -0.5, 1.3)),
            position = position_dodge(0.75), size = 2) +
  scale_fill_viridis_d(option = "viridis") +
  scale_y_continuous(limits = c(-0.05, 1.10)) +
  labs(title = "(A) Model Performance",
       subtitle = paste0("Model performance across different data sovereignty regimes."),
       x = NULL, y = expression(R^2), fill = NULL) +
  theme_minimal() +
  theme(legend.position = 'bottom',
        plot.title = element_text(size = 10, face = "bold"),
        plot.subtitle = element_text(size = 8),
        axis.title.y = element_text(size = 7),
        axis.title.x = element_text(size = 7),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(size = 7),
        axis.text.y = element_text(size = 7),
        axis.line.x  = element_line(size = 0.15),
        axis.line.y  = element_line(size = 0.15))

# ── Plot ────────────────────────────────────────────────────────────────
sources_order <- c('Uganda', 'Zimbabwe', 'Zambia')

federated_result <- federated_results %>%
  mutate(across(c(regime, mode), tolower),
         across(c(client, source), ~str_to_title(str_squish(.))))

transfer_df <- federated_result %>%
  filter(regime == "transfer", mode %in% c("source", "finetuned"))

agg <- transfer_df %>%
  group_by(source, client, mode) %>%
  summarise(mean_r2 = mean(R2), sd_r2 = sd(R2), .groups = "drop") %>%
  mutate(sd_r2 = if_else(is.na(sd_r2), 0, sd_r2))


build_panel <- function(src) {
  targets <- setdiff(sources_order, src)
  agg %>%
    filter(source == src, client %in% c(src, targets)) %>%
    mutate(
      client = factor(client, levels = c(src, targets)),
      bar_label = paste0(str_to_title(client), "\n(",
                         if_else(client == src, "source", "fine-tuned"), ")")
    ) %>%
    arrange(client) %>%
    mutate(bar_label = factor(bar_label, levels = unique(bar_label)))
}

panel_df <- bind_rows(lapply(sources_order, build_panel)) %>%
  mutate(source_label = paste0("Source: ", str_to_title(source)),
         source_label = factor(source_label,
          levels = paste0("Source: ", str_to_title(sources_order))))

transfer_plot <- ggplot(panel_df, aes(x = bar_label, y = mean_r2, fill = client)) +
  geom_col(width = 0.99, linewidth = 0.3) +
  geom_errorbar(aes(ymin = mean_r2 - sd_r2, ymax = mean_r2 + sd_r2),
                width = 0.15) +
  geom_text(aes(label = sprintf("%.2f", mean_r2), y = mean_r2 + sd_r2 + 0.02),
            size = 3) +
  facet_wrap(~ source_label, scales = "free_x") +
  scale_fill_viridis_d(option = "viridis") +
  coord_cartesian(ylim = c(0, 1.0)) +
  labs(title = "Transfer Regime: Source Model Performance and Fine-tuned Recipients",
       subtitle = "A source country's own model, and how each recipient performs after fine-tuning from it.",
       x = NULL, y = expression(R^2)) +
  scale_y_continuous(limits = c(0, 1.0), expand = expansion(mult = c(0, 0.05))) +
  scale_x_discrete(expand = expansion(add = c(0.6, 0.6))) +
  theme_bw(base_size = 12) +
  theme(legend.position = 'none',
        plot.title = element_text(size = 11, face = "bold"),
        plot.subtitle = element_text(size = 9),
        axis.title.y = element_text(size = 8),
        axis.title.x = element_text(size = 8),
        axis.text.x = element_text(size = 8),
        axis.text.y = element_text(size = 8),
        #axis.line.x  = element_line(size = 0.15),
        #axis.line.y  = element_line(size = 0.15),
        legend.title = element_text(size = 9),
        legend.text = element_text(size = 7)) 

shapely_df$client = factor(
  shapely_df$client,
  levels = c('uganda', 'zimbabwe', 'zambia'),
  labels = c('Uganda', 'Zimbabwe', 'Zambia')
)
contribution <- ggplot(shapely_df, aes(x = client, y = shapley, fill = client,
                                       width = 0.99)) +
  geom_col() + geom_hline(yintercept = 0) +
  geom_text(aes(label = round(shapley, 2),
                vjust = ifelse(shapley >= 0, -0.5, 1.3)),
            size = 2.5) +
  labs(title = "(C) Country Contribution",
       subtitle = "Each country's marginal contribution to model Performance.",
       y = expression('Contribution factor '(phi)), x = NULL) +
  theme_minimal() + scale_y_continuous(limits = c(-0.1, 0.4)) +
  scale_fill_viridis_d(option = "viridis", na.value = "grey90") +
  theme(legend.position = 'none',
        plot.title = element_text(size = 10, face = "bold"),
        plot.subtitle = element_text(size = 8),
        axis.title.y = element_text(size = 7),
        axis.title.x = element_text(size = 7),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(size = 7),
        axis.text.y = element_text(size = 7),
        axis.line.x  = element_line(size = 0.15),
        axis.line.y  = element_line(size = 0.15))


# ── Plot ─────────────────────────────────────────────────────────────────
client_order <- c("uganda", "zimbabwe", "zambia")
summary_df1 <- constructs %>%
  group_by(client) %>%
  summarise(
    local_mean = mean(local),
    local_sd   = sd(local),
    gamma_mean = mean(gamma_fedavg),
    gamma_sd   = sd(gamma_fedavg),
    .groups = "drop"
  ) %>%
  mutate(client = factor(client, levels = client_order))
summary_df1$client = factor(
  summary_df1$client,
  levels = c('uganda', 'zimbabwe', 'zambia'),
  labels = c('Uganda', 'Zimbabwe', 'Zambia')
)


model_strength <- ggplot(summary_df1, aes(x = local_mean, y = gamma_mean, color = client)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  geom_errorbar(aes(ymin = gamma_mean - gamma_sd, ymax = gamma_mean + gamma_sd),
                width = 0.01) +
  geom_point(size = 2, stroke = 0.5) +
  geom_text(aes(label = client), hjust = -0.3, vjust = -0.6, size = 3.5,
            show.legend = FALSE) +
  labs(title = "(B) Model Strength",
       subtitle = 'Standalone model strength vs. federation gain.', 
       x = expression("Local (standalone)" (R[local]^2)),
       y = expression("Federation gain " (Gamma * ", FedAvg"))) +
  scale_color_viridis_d(option = "viridis", na.value = "grey90") +
  theme_minimal() + scale_x_continuous(limits = c(0, 1.2)) +
  theme(legend.position = 'none',
        plot.title = element_text(size = 10, face = "bold"),
        plot.subtitle = element_text(size = 8),
        axis.title.y = element_text(size = 8),
        axis.title.x = element_text(size = 8),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(size = 7),
        axis.text.y = element_text(size = 7),
        axis.line.x  = element_line(size = 0.15),
        axis.line.y  = element_line(size = 0.15))


costs <- ggarrange(mmd_plot, sovereignty, federation_gain,
  ncol = 3,font.label = list(size = 9))

bottom_row <- ggarrange(model_strength, contribution, 
                   ncol = 2,font.label = list(size = 9))

all <- ggarrange(model_regimes, bottom_row, 
                 nrow = 2,font.label = list(size = 9))

dir.create(file.path(folder, 'figures'), showWarnings = FALSE)
path = file.path(folder, 'figures', 'costs.png')
png(path, units = "in", width = 8.5, height = 3, res = 480)
print(costs)
dev.off()

dir.create(file.path(folder, 'figures'), showWarnings = FALSE)
path = file.path(folder, 'figures', 'performance.png')
png(path, units = "in", width = 7, height = 7, res = 480)
print(all)
dev.off()

