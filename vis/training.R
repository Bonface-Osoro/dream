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

###########################
## 4. DENSITY PLOTS LSTM ##
###########################
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

#############################
## 5. ERROR PLOTS XG BOOST ##
#############################
df4 <- read.csv(file.path(folder, '..', 'results', 'final',
   'zimbabwe_validation',  'comparison_per_location.csv'))


r2_base  <- round(max(df4$R2_baseline,  na.rm = TRUE), 4)
r2_ft    <- round(max(df4$R2_finetuned, na.rm = TRUE), 4)
rmse_base <- round(mean(df4$RMSE_baseline,  na.rm = TRUE), 4)
rmse_ft   <- round(mean(df4$RMSE_finetuned, na.rm = TRUE), 4)

df_long4 <- bind_rows(
  data.frame(
    observed  = df4$observed_baseline,
    predicted = df4$predicted_baseline,
    approach  = paste0("Baseline (Uganda only) \nR\u00b2 = ", r2_base, "  RMSE = ", rmse_base)
  ),
  data.frame(
    observed  = df4$observed_finetuned,
    predicted = df4$predicted_finetuned,
    approach  = paste0("Fine-tuned (warm-start) \nR\u00b2 = ", r2_ft, "  RMSE = ", rmse_ft)
  )
)

df_long4$approach <- factor(df_long4$approach, levels = unique(df_long4$approach))

panel_colors <- c("#E07B39", "#2176AE")
names(panel_colors) <- levels(df_long4$approach)

xg_res_plot <- ggplot(df_long4, aes(x = observed, y = predicted, colour = approach)) +
  geom_point(alpha = 0.25, size = 0.8) + 
  geom_abline(slope = 1, intercept = 0, colour = "red",
              linetype = "dashed", linewidth = 0.8) +
  scale_colour_manual(values = panel_colors, guide = "none") +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  facet_wrap(~ approach, ncol = 2) +
  labs(title = "(A) XG Boost Model", 
       subtitle = "MRI XG Boost baseline vs fine-tuned (Warm-Start Boosting) for \n2020-2022 test period", 
       x = "Observed MRI", y = "Predicted MRI") +
  theme_bw(base_size = 12) +
  theme(legend.position = 'bottom',
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

##############################
## 6. SERIES PLOTS XG BOOST ##
##############################
df6 <- read.csv(file.path(folder, '..', 'results', 'final',
                         'zimbabwe_validation',  'comparison_per_time.csv')) %>% 
  arrange(year, month_num) %>% 
  mutate(time_step = row_number() - 1)

df6 <- df6 %>% arrange(year, month_num) %>% mutate(time_step = row_number() - 1)

# Reshape to long format — one row per line per time step
df_long6 <- bind_rows(
  data.frame(time_step = df6$time_step, mean_mri = df6$observed_baseline,  
             series = "Observed"),
  data.frame(time_step = df6$time_step, mean_mri = df6$predicted_baseline,  
             series = "Baseline"),
  data.frame(time_step = df6$time_step, mean_mri = df6$predicted_finetuned, 
             series = "Fine-tuned")
)

df_long6$series <- factor(df_long6$series, levels = c("Observed", "Baseline", 
                                                      "Fine-tuned"))

temp_pred <- ggplot(df_long6, aes(x = time_step, y = mean_mri,
  colour = series, linetype = series, linewidth = series)) + geom_line() +
  scale_colour_manual(name = NULL, values = c("Observed" = "black", 
                      "Baseline" = "#E07B39", "Fine-tuned" = "#2196C4")) +
  scale_linetype_manual(name = NULL, values = c("Observed" = "solid", 
                      "Baseline" = "dashed", "Fine-tuned" = "dashed")) +
  scale_linewidth_manual(name = NULL,
    values = c("Observed" = 0.8, "Baseline" = 0.6, "Fine-tuned" = 0.6)) +
  scale_x_continuous(breaks = seq(0, max(df6$time_step), by = 5)) +
  labs(title = "(B) Temporal Trend Comparison",
       subtitle = "Mean predicted MRI across test locations by time step for Zimbabwe \n(2020–2022)",
    x = "Time step", y = "Mean MRI") + theme_minimal() +
  theme(legend.position = 'bottom',
        plot.title = element_text(size = 11, face = "bold"),
        plot.subtitle = element_text(size = 9),
        axis.title.y = element_text(size = 8),
        axis.title.x = element_text(size = 8),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(size = 8),
        axis.text.y = element_text(size = 8),
        axis.line.x  = element_line(size = 0.15),
        axis.line.y  = element_line(size = 0.15),
        legend.title = element_text(size = 8),
        legend.text = element_text(size = 7))

#############################
## 7. ERROR PLOTS XG BOOST ##
#############################
df5 <- read.csv(file.path(folder, '..', 'results', 'final',
      'zimbabwe_validation',  'lstm_comparative', 'comparison_per_location.csv'))

strategies <- list(list(label = "Baseline", obs = "observed_baseline", 
  pred = "predicted_baseline", r2 = "R2_baseline", rmse = "RMSE_baseline"),
  list(label = "Freeze All LSTM", obs = "observed_freeze_all_lstm", 
      pred = "predicted_freeze_all_lstm", r2 = "R2_freeze_all_lstm",  
      rmse = "RMSE_freeze_all_lstm"), list(label = "Freeze LSTM L0", 
      obs = "observed_freeze_lstm_l0", pred = "predicted_freeze_lstm_l0",
      r2 = "R2_freeze_lstm_l0",   rmse = "RMSE_freeze_lstm_l0"),
  list(label = "Freeze LSTM L1", obs = "observed_freeze_lstm_l1", 
      pred = "predicted_freeze_lstm_l1", r2 = "R2_freeze_lstm_l1",   
      rmse = "RMSE_freeze_lstm_l1"), list(label = "Retrain All", 
      obs = "observed_retrain_all", pred = "predicted_retrain_all",        
      r2 = "R2_retrain_all",      rmse = "RMSE_retrain_all"))

df_long5 <- bind_rows(lapply(strategies, function(s) {
  r2   <- round(max(df5[[s$r2]],   na.rm = TRUE), 3)
  rmse <- round(max(df5[[s$rmse]], na.rm = TRUE), 3)
  data.frame(
    observed  = df5[[s$obs]],
    predicted = df5[[s$pred]],
    approach  = paste0(s$label, "\nR\u00b2 = ", r2, "  RMSE = ", rmse)
  )
}))

df_long5$approach <- factor(df_long5$approach, levels = unique(df_long5$approach))
panel_colors <- c("#E07070", "#F0A060", "#9B8DC0", "#5BA8A0", "#6FA8D0")
names(panel_colors) <- levels(df_long5$approach)

lstm_res_plot <- ggplot(df_long5, aes(x = observed, y = predicted, colour = approach)) +
  geom_point(alpha = 0.25, size = 0.6) +
  geom_abline(slope = 1, intercept = 0, colour = "red",
              linetype = "dashed", linewidth = 0.7) +
  scale_colour_manual(values = panel_colors, guide = "none") +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  facet_wrap(~ approach, ncol = 5) +
  labs(title = "(C) LSTM Model", 
       subtitle = "MRI modified LSTM prediction results for 2020-2022 test period (Zimbabwe)", 
    x = "Observed MRI", y = "Predicted MRI") +
  theme_bw(base_size = 12) +
  theme(legend.position = 'bottom',
        plot.title = element_text(size = 11, face = "bold"),
        plot.subtitle = element_text(size = 9),
        axis.title.y = element_text(size = 8),
        axis.title.x = element_text(size = 8),
        axis.text.x = element_text(size = 8),
        axis.text.y = element_text(size = 8),
        legend.title = element_text(size = 9),
        legend.text = element_text(size = 7)) 
  

top_row <- ggarrange(xg_res_plot, temp_pred,
                     ncol = 2, nrow = 1)

combined_plots_1 <- ggarrange(top_row, lstm_res_plot,
                            ncol = 1, nrow = 2,
                            heights = c(0.9, 0.7))

path = file.path(folder, 'figures', 'training_plots_1.png')
png(path, units="in", width=9, height=6.5, res=300)
print(combined_plots_1)
dev.off()




