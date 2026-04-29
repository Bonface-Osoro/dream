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
## UGA MRI MAPS ##
##################
national_shp = st_read(file.path(folder, '..', 'data', 'raw', 'shapefiles', 
                                 'uga.shp'))
ug_data <- st_read(file.path(folder, '..', 'data', 'raw', 'shapefiles', 
                             'gadm41_UGA_2.shp'))
mri_data <- read.csv(file.path(folder, '..', 'results', 'final', 
                               'region_2_year_mri.csv'))
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

##################
## ZWE MRI MAPS ##
##################
nation_shp = st_read(file.path(folder, '..', 'data', 'raw', 'shapefiles', 
                                 'gadm41_ZWE_0.shp'))
zw_data <- st_read(file.path(folder, '..', 'data', 'raw', 'shapefiles', 
                             'gadm41_ZWE_2.shp'))
zw_mri_data <- read.csv(file.path(folder, '..', 'results', 'final', 
                               'ZWE_region_2_year_mri.csv'))
combined <- zw_data %>%
  left_join(zw_mri_data, by = "GID_2")

combined <- combined %>%
  filter(!is.na(year) & year >= 2014 & year <= 2024)

zwe_annual_mri <- ggplot() +
  geom_sf(data = nation_shp, fill = NA, color = "black", size = 0.01) +
  geom_sf(data = combined, aes(fill = mri_value), color = 'NA') +
  scale_fill_viridis_c(
    option    = "viridis",
    na.value  = "grey90",
    limits    = c(0.20, 0.97),        # clips flat extremes
    oob       = scales::squish,       # values outside limits squeezed to edge colour
    trans     = "sqrt",               # stretches low-risk end visually
    breaks    = c(0.2, 0.4, 0.6, 0.75, 0.85, 0.95),
    labels    = c("0.2", "0.4", "0.6", "0.75", "0.85", "0.95")
  )  +
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

path = file.path(folder, 'figures', '2009_2020_annual_mri.png')
png(path, units="in", width=7, height=6, res=720)
print(annual_mri)
dev.off()




