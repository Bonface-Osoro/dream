library(ggplot2)
library(dplyr)
library(tidyr)

suppressMessages(library(tidyverse))
folder <- dirname(rstudioapi::getSourceEditorContext()$path)


##############################
## 1. MODEL TRAINING CURVES ##
##############################

xgb  <- read.csv(file.path(folder, '..', 'results', 'final',
                           'xgboost', 'xgb_training_log.csv'))

lstm <- read.csv(file.path(folder, '..', 'results', 'final',
                           'lstm', 'lstm_training_log.csv'))

xgb_long <- xgb %>%
  rename(step = Iteration) %>%
  pivot_longer(cols = c(Train_RMSE, Val_RMSE),
               names_to  = "series",
               values_to = "RMSE") %>%
  mutate(series = recode(series,
                         Train_RMSE = "Train RMSE",
                         Val_RMSE   = "Val RMSE"),
         model = "XGBoost")

lstm_long <- lstm %>%
  rename(step = Epoch) %>%
  mutate(Train_RMSE = sqrt(Train_Loss),
         Val_RMSE   = sqrt(Val_Loss)) %>%
  select(step, Train_RMSE, Val_RMSE) %>%
  pivot_longer(cols = c(Train_RMSE, Val_RMSE),
               names_to  = "series",
               values_to = "RMSE") %>%
  mutate(series = recode(series,
                         Train_RMSE = "Train RMSE",
                         Val_RMSE   = "Val RMSE"),
         model = "LSTM")

df <- bind_rows(xgb_long, lstm_long)
df$model <- factor(df$model, levels = c("XGBoost", "LSTM"))


train_plots <- ggplot(df, aes(x = step, y = RMSE, colour = series)) +
  geom_point(size = 0.6, alpha = 0.8) +
  geom_line(linewidth = 0.6, alpha = 0.6) +
  scale_colour_manual(name   = NULL,
    values = c("Train RMSE" = "#2176AE", "Val RMSE" = "#E07B39")) +
  scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0.02, 0.05))) +
  scale_x_continuous(expand = expansion(mult = c(0.01, 0.02))) +
  facet_wrap(~ model, ncol = 2, scales = "free") +
  labs(title = "(A) Model Training Curves",
       subtitle = "Learning curves showing model convergence during training on Uganda data",
    x = "Iteration / Epoch", y = "RMSE") +
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
        axis.line.y  = element_line(size = 0.15),
        legend.title = element_text(size = 7),
        legend.text = element_text(size = 6)) 

###########################
## 2. MODEL TEST METRICS ##
###########################
xgb_metrics  <- read.csv(file.path(folder, '..', 'results', 'final', 'xgboost',
                                   'xg_test_metrics.csv'))  %>%
  filter(Metric %in% c("R2", "RMSE", "MAE")) %>%
  mutate(model = "XGBoost")

lstm_metrics <- read.csv(file.path(folder, '..', 'results', 'final', 'lstm', 
                                   'lstm_test_metrics.csv')) %>%
  filter(Metric %in% c("R2", "RMSE", "MAE")) %>%
  mutate(Value = as.numeric(Value), model = "LSTM")

xgb_mse <- read.csv(file.path(folder, '..', 'results', 'final', 'xgboost',
                              'xg_test_metrics.csv')) %>%
  filter(Metric == "MSE") %>% pull(Value)

xgb_metrics <- bind_rows(
  xgb_metrics,
  data.frame(Metric = "RMSE", Value = sqrt(xgb_mse), model = "XGBoost")
) %>% filter(Metric != "MSE")

lstm_mse <- read.csv(file.path(folder, '..', 'results', 'final', 'lstm', 
                               'lstm_test_metrics.csv')) %>%
  filter(Metric == "MSE") %>% 
  mutate(Value = as.numeric(Value)) %>%
  pull(Value)

lstm_metrics <- bind_rows(
  lstm_metrics,
  data.frame(Metric = "RMSE", Value = sqrt(lstm_mse), model = "LSTM")
) %>% filter(Metric != "MSE")

# ── Combine ────────────────────────────────────────────────────────────────────
df <- bind_rows(xgb_metrics, lstm_metrics)
df$model  <- factor(df$model,  levels = c("XGBoost", "LSTM"))
df$Metric <- factor(df$Metric, levels = c("R2", "RMSE", "MAE"))


test_metrics <- ggplot(df, aes(x = Metric, y = Value, fill = Metric)) +
  geom_col(width = 0.98, alpha = 0.85) +
  geom_text(aes(label = round(Value, 4)),
            vjust = -0.4, size = 3.2) +
  scale_fill_manual(
    values = c("R2" = "#2176AE", "RMSE" = "#E07B39", "MAE" = "#57A773"),
    guide  = "none") +
  scale_y_continuous(limits = c(0, 1.08),
                     expand = expansion(mult = c(0.0, 0.05))) +
  facet_wrap(~ model, ncol = 2, scales = "free_y") +
  labs(title = "Model Test Performance",
    subtitle = "R², RMSE and MAE evaluated on the Uganda held-out test set",
    x = NULL,
    y = "Value") + theme_minimal() +
  theme(legend.position = 'bottom',
        plot.title = element_text(size = 12, face = "bold"),
        plot.subtitle = element_text(size = 10),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(size = 9),
        axis.text.y = element_text(size = 9),
        axis.title.x = element_text(size = 8),
        axis.line.x  = element_line(size = 0.15),
        axis.line.y  = element_line(size = 0.15),
        legend.title = element_text(size = 8),
        legend.text = element_text(size = 7))















