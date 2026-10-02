library(ggplot2)
library(dplyr)
library(lubridate)
library(patchwork)
library(scales)
library(fields) 
library(sf)

suppressMessages(library(tidyverse))
folder <- dirname(rstudioapi::getSourceEditorContext()$path)
######################
## MRI TVALIDATION  ##
######################
validation <- read.csv(file.path(folder, '..', 'results', 'final', 'UGA_dhs',
                                 'UGA_mri_validation_results.csv'))

category_levels <- c("extremely low risk", "low risk", "high risk",
                     "extremely high risk")
category_labels <- c("Extremely\nlow", "Low", "High", "Extremely\nhigh")

validation <- validation %>%
  mutate(risk_category = factor(risk_category, levels = category_levels))

uga_positivity <- ggplot(validation, aes(x = risk_category, y = positivity_pct, 
                      fill = risk_category, width = 0.99)) + geom_col() +
  geom_errorbar(aes(ymin = ci_lower_pct, ymax = ci_upper_pct),
                width = 0.15, linewidth = 0.6, colour = "red") +
  geom_text(aes(label = sprintf("%.1f%%", positivity_pct)),
            vjust = -2.6, size = 3.6) +
  scale_fill_viridis_d(option = "viridis", na.value = "grey90") +
  scale_x_discrete(labels = category_labels) +
  scale_y_continuous(limits = c(0, 55), expand = expansion(mult = c(0, 0.05)),
                     labels = function(x) paste0(x, "%")) +
  labs(title = "Malaria Test Positivity Rate",
    subtitle = "RDT positivity calculated as a proportion of individuals with confirmed malaria as a percentage of all individuals tested within the predicted risk category.",
    x = "Risk category", y = "Rapid Diagnostic Test (RDT) positivity (%)") + theme_minimal() +
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

#"Children classified as extremely high risk had 3.57 times the odds of 
#RDT-confirmed malaria compared to those classified as extremely low risk 
#(46.1% vs. 12.9% positivity, 95% CI 45.7-46.5% vs. 12.6-13.2%).


uga_risk_ratio <- ggplot(validation, aes(x = risk_category, y = risk_ratio_vs_lowest, 
                                         fill = risk_category, width = 0.99)) + geom_col() +
  geom_text(aes(label = sprintf("%.1f%%", risk_ratio_vs_lowest)),
            vjust = -2.6, size = 3.6) +
  scale_fill_viridis_d(option = "viridis", na.value = "grey90") +
  scale_x_discrete(labels = category_labels) +
  scale_y_continuous(limits = c(0, 5), expand = expansion(mult = c(0, 0.05)),
                     labels = function(x) paste0(x, "%")) +
  labs(title = "(B) Malaria Risk Ratio",
       subtitle = "Estimated by dividing the RDT positivity in each risk category by the RDT positivity in the low-risk reference category.",
       x = "Risk category", y = "Malaria Risk Ratio") + theme_minimal() +
  scale_x_discrete(labels = category_labels, expand = expansion(mult = c(0, 0)))+
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

##################
## UGA MRI MAPS ##
##################
national_shp = st_read(file.path(folder, '..', 'data', 'raw', 'shapefiles', 
                                 'uga.shp'))
ug_data <- st_read(file.path(folder, '..', 'data', 'raw', 'shapefiles', 
                             'gadm41_UGA_2.shp'))
mri_data <- read.csv(file.path(folder, '..', 'results', 'final','UGA_dhs',
                               'UGA_GID2_dhs_risk_scores.csv'))
combined <- ug_data %>%
  left_join(mri_data, by = "GID_2")

combined <- combined %>%
  filter(!is.na(year) & year >= 2009 & year <= 2020)

annual_mri <- ggplot() +
  geom_sf(data = national_shp, fill = NA, color = "black", size = 0.01) +
  geom_sf(data = combined, aes(fill = mean_mri), color = 'NA') +
  scale_fill_viridis_c(option = "viridis", na.value = "grey90") +
  facet_wrap(~ year) +
  labs(title = "Malaria Risk Index (MRI)",
       subtitle = "MRI derived using principal component analysis (PCA) as a linear combination of net access, net use, parasite, incidence \nand mortality rates.",
       fill = "MRI Value") +
  theme(legend.position = 'bottom',
        plot.margin = margin(0, 0, 0, 0),              
        plot.title = element_text(size = 9, face = "bold"),
        plot.subtitle = element_text(size = 7),
        axis.title.y = element_text(size = 7),
        axis.title.x = element_text(size = 7),
        panel.border = element_blank(),
        strip.text = element_text(size = 5),
        axis.text.x = element_text(size = 5),
        axis.text.y = element_text(size = 5),
        axis.line.x  = element_line(size = 0.15),
        axis.line.y  = element_line(size = 0.15),
        legend.title = element_text(size = 6),
        legend.text = element_text(size = 5)) 







