library(ggplot2)
library(tidyverse)
library(ggtext)
library(sf)
library(RColorBrewer)
library(gt)
library(readr)
library(patchwork)
library(ggplotify)
library(gridExtra)
library(readxl)
library(haven)
library(ggpubr)

suppressMessages(library(tidyverse))
folder <- dirname(rstudioapi::getSourceEditorContext()$path)

##################
## ZWE MRI MAPS ##
##################
national_shp = st_read(file.path(folder, '..', 'data', 'raw', 'shapefiles', 
                                 'gadm41_ZWE_0.shp'))
ug_data <- st_read(file.path(folder, '..', 'data', 'raw', 'shapefiles', 
                             'gadm41_ZWE_2.shp'))
mri_data <- read.csv(file.path(folder, '..', 'results', 'final', 
                               'ZWE_region_2_monthly_mri.csv'))
combined <- ug_data %>%
  left_join(mri_data, by = "GID_2")

monthly_mri <- ggplot() +
  geom_sf(data = national_shp, fill = NA, color = "black", size = 0.01) +
  geom_sf(data = combined, aes(fill = mean_mri), color = 'NA') +
  scale_fill_viridis_c(option = "viridis", na.value = "grey90") +
  facet_wrap(~ month) +
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
