library(data.table)
library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(reshape2)
library(ggbeeswarm)
library(viridis)
library(ggpubr)

suppressMessages(library(tidyverse))
folder <- dirname(rstudioapi::getSourceEditorContext()$path)

#############################
## 1. XGBOOST SHAP SUMMARY ##
#############################
shap_file <- read.csv(file.path(folder, '..', 'results', 'final', 
                               'xgboost', 'lagged_shap_values_full.csv'))

shap_cols <- c("ndvi", "precipitation_mm", "temperature_C",
               "elevation_m", "month_sin", "month_cos",
               "longitude", "latitude")

readable_names <- c(ndvi = "NDVI", precipitation_mm = "Precipitation (mm)",
  temperature_C = "Temperature (°C)", elevation_m = "Elevation (m)",
  month_sin = "Seasonality (sin)", month_cos = "Seasonality (cos)",
  longitude = "Longitude", latitude = "Latitude")

value_cols <- paste0(shap_cols, "_value")
shap_long <- reshape(shap_file, varying = shap_cols, v.names = "shap_value",
  timevar = "feature", times = shap_cols, direction = "long")

value_long <- reshape(shap_file, varying = value_cols, v.names = "feature_value",
  timevar = "feature", times = shap_cols, direction = "long")

shap_long$feature_value <- value_long$feature_value
shap_long$feature <- readable_names[shap_long$feature]

feature_order <- shap_long %>% group_by(feature) %>%
  summarise(importance = mean(abs(shap_value))) %>%arrange(importance)

shap_long$feature <- factor(shap_long$feature, levels = feature_order$feature)

xg_shap_summary <- ggplot(shap_long, aes(x = shap_value, y = feature, 
                                      color = feature_value)) +
  geom_quasirandom(groupOnX = FALSE, alpha = 0.5, size = 0.8) +
  scale_color_viridis(option = "viridis", direction = 1) +
  theme_minimal() +
  labs(title = "SHAP Summary Plot", 
       subtitle = '(a) XGBoost Model', x = 'SHAP value (impact on model output)',
    y = "", color = 'Feature value') +
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

importance_df <- shap_long %>%
  group_by(feature) %>%summarise(mean_abs_shap = mean(abs(shap_value))) %>%
  arrange(mean_abs_shap)

xg_shap_bar <- ggplot(importance_df, aes(x = reorder(feature, mean_abs_shap),
                                           y = mean_abs_shap, fill = mean_abs_shap)) + geom_col() +coord_flip() +
  scale_fill_viridis(option = "viridis") + theme_minimal() +
  labs(title = 'SHAP Feature Importance', subtitle = '(a) XGBoost Model', x = '',
       y = 'Mean SHAP value') +
  geom_text(aes(label = sprintf("+%.02f", mean_abs_shap)),
            hjust = -0.1, size = 2) +
  theme(legend.position = 'none',
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
        legend.text = element_text(size = 7)) +
  scale_y_continuous(expand = c(0, 0)) +
  expand_limits(y = max(importance_df$mean_abs_shap) * 1.5)


##########################
## 2. LSTM SHAP SUMMARY ##
##########################
lstm_file <- read.csv(file.path(folder, '..', 'results', 'final', 
                                'lstm', 'lstm_shap_values.csv'))
shap_cols <- c("ndvi", "precipitation_mm", "temperature_C",
               "elevation_m", "month_sin", "month_cos", "mri_lag1")

readable_names <- c(ndvi = "NDVI", precipitation_mm = "Precipitation (mm)",
    temperature_C = "Temperature (°C)", elevation_m = "Elevation (m)",
    month_sin = "Seasonality (sin)", month_cos = "Seasonality (cos)",
    mri_lag1 = "MRI Lags")

value_cols <- paste0(shap_cols, "_value")
shap_long <- reshape(lstm_file, varying = shap_cols, v.names = "shap_value",
      timevar = "feature", times = shap_cols, direction = "long")

value_long <- reshape(lstm_file, varying = value_cols, v.names = "feature_value",
                      timevar = "feature", times = shap_cols, direction = "long")

shap_long$feature_value <- value_long$feature_value
shap_long$feature <- readable_names[shap_long$feature]

feature_order <- shap_long %>% group_by(feature) %>%
  summarise(importance = mean(abs(shap_value))) %>%arrange(importance)

shap_long$feature <- factor(shap_long$feature, levels = feature_order$feature)

lstm_shap_summary <- ggplot(shap_long, aes(x = shap_value, y = feature, 
                                      color = feature_value)) +
  geom_quasirandom(groupOnX = FALSE, alpha = 0.5, size = 0.8) +
  scale_color_viridis(option = "viridis", direction = 1) +
  theme_minimal() + labs(title = ' ', subtitle = '(b) LSTM Model',
       x = 'SHAP value (impact on model output)',
       y = "", color = 'Feature value') +
  theme(legend.position = 'bottom',
        plot.title = element_text(size = 12, face = "bold"),
        plot.subtitle = element_text(size = 10),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(size = 9),
        axis.title.x = element_text(size = 8),
        axis.text.y = element_text(size = 9),
        axis.line.x  = element_line(size = 0.15),
        axis.line.y  = element_line(size = 0.15),
        legend.title = element_text(size = 8),
        legend.text = element_text(size = 7))


importance_df <- shap_long %>%
  group_by(feature) %>%summarise(mean_abs_shap = mean(abs(shap_value))) %>%
  arrange(mean_abs_shap)

lstm_shap_bar <- ggplot(importance_df, aes(x = reorder(feature, mean_abs_shap),
      y = mean_abs_shap, fill = mean_abs_shap)) + geom_col() +coord_flip() +
  scale_fill_viridis(option = "viridis") + theme_minimal() +
  labs(title = ' ', subtitle = '(b) LSTM Model', x = "", y = "Mean SHAP value") +
  geom_text(aes(label = sprintf("+%.02f", mean_abs_shap)),
            hjust = -0.1, size = 2) +
  theme(legend.position = 'none',
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
        legend.text = element_text(size = 7)) +
  scale_y_continuous(expand = c(0, 0)) +
  expand_limits(y = max(importance_df$mean_abs_shap) * 1.5)

summary_plots <- ggarrange(xg_shap_summary, lstm_shap_summary, ncol = 2,
                               nrow = 1, heights = c(1, 1))


bar_plots <- ggarrange(xg_shap_bar, lstm_shap_bar, ncol = 2,
                           nrow = 1, heights = c(1, 1))

combined_plots <- ggarrange(summary_plots, bar_plots, ncol = 1,
                            nrow = 2)

path = file.path(folder, 'figures', 'explainable_ai.png')
png(path, units="in", width=7, height=6, res=300)
print(combined_plots)
dev.off()


lstm_mri <- read.csv(file.path(folder, '..', 'results', 'final','lstm', 
                              'lstm_per_location_metrics.csv'))

df_hold <- lstm_mri %>%
  filter(year == 2020, month_num >= 6) %>%
  left_join(metrics, by = c("longitude", "latitude"))

obs <- df_hold$monthly_mri



