library(ggplot2)
library(dplyr)

suppressMessages(library(tidyverse))
folder <- dirname(rstudioapi::getSourceEditorContext()$path)

##########################
## 1. XG BOOST STRATEGY ##
##########################
df1 <- read.csv(file.path(folder, '..', 'results', 'final',
     'zimbabwe_validation', 'finetuning_search.csv'))

df_long <- df1 %>%
  select(n_trees, val_R2, val_RMSE) %>%
  pivot_longer(cols = c(val_R2, val_RMSE),
     names_to  = "metric", values_to = "value") %>%
  mutate(metric = recode(metric,
     val_R2   = "Validation R\u00b2", val_RMSE = "Validation RMSE"))

xg_boost_val <- ggplot(df_long, aes(x = n_trees, y = value, colour = metric, shape = metric)) +
  geom_line(linewidth = 0.3) + theme_minimal() +
  scale_colour_manual(name   = NULL,
    values = c("Validation R\u00b2" = "#1f77b4", "Validation RMSE" = "#ff7f0e")) +
  scale_shape_manual(name   = NULL,
    values = c("Validation R\u00b2" = 16, "Validation RMSE" = 15)) +
  labs(title = "(A) XGBoost Residual Booster",
       subtitle = "Learning Curve at different number of residual \ntrees in the XG Boost model.",
    x = "Number of residual trees", y = "Value") +
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
        axis.line.y  = element_line(size = 0.15),
        legend.title = element_text(size = 7),
        legend.text = element_text(size = 6)) 


###############################
## 2. LSTM TRAINING STRATEGY ##
###############################
df <- read.csv(file.path(folder, '..', 'results', 'final',
   'zimbabwe_validation', 'lstm_comparative', 'finetuning_training_logs.csv'))

df <- df %>%
  mutate(strategy = recode(strategy, freeze_all_lstm = "Freeze All LSTM",
  freeze_lstm_l0  = "Freeze LSTM L0", freeze_lstm_l1  = "Freeze LSTM L1",
  retrain_all     = "Retrain All" ))

df$strategy <- factor(df$strategy,
 levels = c("Freeze All LSTM", "Freeze LSTM L0", "Freeze LSTM L1", "Retrain All"))

strategy_colors <- c("Freeze All LSTM" = "#D95F02", 
  "Freeze LSTM L0"  = "#E6AB02", "Freeze LSTM L1"  = "#1B9E77",   
  "Retrain All"     = "#1F78B4")

lstm_val <- ggplot(df, aes(x = epoch, y = val_loss, colour = strategy)) +
  geom_line(linewidth = 0.3, alpha = 0.9) + theme_minimal() +
  scale_colour_manual(values = strategy_colors, name = NULL) +
  guides(fill   = guide_legend(nrow = 2),
         colour = guide_legend(nrow = 2)) +
  labs(title = "(B) LSTM Fine-tuning",
    subtitle = "Validation Loss by training Strategy",
    x = "Epochs", y = "Validation loss (MSE)") +
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
    axis.line.y  = element_line(size = 0.15),
    legend.title = element_text(size = 7),
    legend.text = element_text(size = 6)) + 
  scale_y_continuous(expand = c(0, 0),
     labels = function(y) format(y, scientific = FALSE), limits = c(0, 0.019)) +
  scale_x_continuous(expand = c(0, 0),
     limits = c(min(df$epoch, epoch = TRUE), 15),
     breaks = seq(min(df$epoch, na.rm = TRUE), 15, by = 5))
###############################
## 3. DENSITY PLOTS XG Boost ##
###############################
df3 <- read.csv(file.path(folder, '..', 'results', 'final',
       'zimbabwe_validation', 'comparison_per_location.csv'))

df_long3 <- data.frame(value = c(df3$RMSE_baseline, df3$RMSE_finetuned),
  strategy = c(rep("Baseline", nrow(df3)), rep("XGBoost Residual", nrow(df3))))

df_long3$strategy <- factor(df_long3$strategy,
    levels = c("Baseline", "XGBoost Residual"))

strategy_colors <- c("Baseline" = "#E07070",
  "XGBoost Residual"= "#5BA8A0")

XG_error <- ggplot(df_long3, aes(x = value, colour = strategy, fill = strategy)) +
  geom_density(alpha = 0.45, linewidth = 0) + theme_minimal() +
  geom_vline(xintercept = 0, colour = "grey40", linewidth = 0.5, linetype = "dashed") +
  scale_colour_manual(name = NULL, values = strategy_colors) +
  scale_fill_manual(  name = NULL, values = strategy_colors) +
  labs(title = "(C) XG Boost Error",
       subtitle = "Error distributions grouped by different training Strategies",
       y = "Density", x = "RMSE per location") +
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
        axis.line.y  = element_line(size = 0.15),
        legend.title = element_text(size = 7),
        legend.text = element_text(size = 6)) +
  scale_x_continuous(expand = c(0, 0), limits = c(0, NA)) + 
  scale_y_continuous(expand = c(0, 0),
    labels = function(y) format(y, scientific = FALSE), limits = c(0, 50))

######################
## 3. DENSITY PLOTS ##
######################
df2 <- read.csv(file.path(folder, '..', 'results', 'final',
   'zimbabwe_validation', 'lstm_comparative', 'comparison_per_location.csv'))

df_long1 <- data.frame(value    = c(
  df2$RMSE_baseline, df2$RMSE_freeze_lstm_l1,
  df2$RMSE_retrain_all, df2$RMSE_freeze_lstm_l0,
  df2$RMSE_freeze_all_lstm),
  strategy = c(rep("Baseline", nrow(df2)), rep("Freeze LSTM L1",  nrow(df2)),
    rep("Retrain All", nrow(df2)), rep("Freeze LSTM L0", nrow(df2)),
    rep("Freeze All LSTM", nrow(df2))))

df_long1$strategy <- factor(df_long1$strategy,
   levels = c("Baseline", "Freeze All LSTM", "Freeze LSTM L0", "Freeze LSTM L1", 
              "Retrain All"))

strategy_colors <- c("Baseline" = "#E07070", "Freeze All LSTM"= "#F0A060",
  "Freeze LSTM L0" = "#9B8DC0", "Freeze LSTM L1" = "#5BA8A0",
  "Retrain All"    = "#6FA8D0")

lstm_error <- ggplot(df_long1, aes(x = value, colour = strategy, fill = strategy)) +
  geom_density(alpha = 0.45, linewidth = 0) +
  geom_vline(xintercept = 0, colour = "grey40", linewidth = 0.5, 
             linetype = "dashed") + theme_minimal() +
  scale_colour_manual(name = NULL, values = strategy_colors) +
  scale_fill_manual(  name = NULL, values = strategy_colors) +
  guides(fill   = guide_legend(nrow = 2),
         colour = guide_legend(nrow = 2)) +
  labs(title = "(D) LSTM Error",
       subtitle = "Error distributions grouped by different training Strategies",
       y = "Density", x = "RMSE per location") +
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
        axis.line.y  = element_line(size = 0.15),
        legend.title = element_text(size = 7),
        legend.text = element_text(size = 6)) +
  scale_x_continuous(expand = c(0, 0), limits = c(0, NA)) + 
  scale_y_continuous(expand = c(0, 0),
      labels = function(y) format(y, scientific = FALSE), limits = c(0, 22)) 


df_long3$model <- "XGBoost"
df_long1$model <- "LSTM"

# Combine — keep only strategies that exist in both or handle separately
df_combined <- bind_rows(df_long1, df_long3)
df_combined$model <- factor(df_combined$model, levels = c("XGBoost", "LSTM"))

# Unified colour palette covering all strategies
strategy_colors <- c(
  "Baseline"        = "#E07070",
  "Freeze All LSTM" = "#F0A060",
  "Freeze LSTM L0"  = "#9B8DC0",
  "Freeze LSTM L1"  = "#5BA8A0",
  "Retrain All"     = "#6FA8D0",
  "XGBoost Residual"= "#3A7DBF"
)


ggplot(df_combined, aes(x = value, colour = strategy, fill = strategy)) +
  geom_density(alpha = 0.45, linewidth = 0.7) + 
  geom_vline(xintercept = 0, colour = "grey40", linewidth = 0.5, linetype = "dashed") +
  scale_colour_manual(name = NULL, values = strategy_colors) +
  scale_fill_manual(  name = NULL, values = strategy_colors) +
  facet_wrap(~ model, ncol = 2) +theme_bw(base_size = 12) +
  theme(legend.position = 'bottom',
        plot.title = element_text(size = 11, face = "bold"),
        plot.subtitle = element_text(size = 9),
        axis.title.y = element_text(size = 8),
        axis.title.x = element_text(size = 8),
        axis.text.x = element_text(size = 8),
        axis.text.y = element_text(size = 8),
        axis.line.x  = element_line(size = 0.15),
        axis.line.y  = element_line(size = 0.15),
        legend.title = element_text(size = 9),
        legend.text = element_text(size = 7)) 

combined_plots <- ggarrange(xg_boost_val, lstm_val, XG_error, lstm_error, 
    ncol = 2, nrow = 2)


path = file.path(folder, 'figures', 'training_plots.png')
png(path, units="in", width=7, height=6.5, res=300)
print(combined_plots)
dev.off()
