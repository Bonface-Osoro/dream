library(ggpubr)
library(tidyverse)
library(sf)
library(gganimate)
library(transformr)

suppressMessages(library(tidyverse))
folder <- dirname(rstudioapi::getSourceEditorContext()$path)
##################
## UGA MRI MAPS ##
##################
national_shp_uga <- st_read(file.path(folder, '..', 'data', 'raw', 'shapefiles',
                                      'uga.shp'))
ug_data <- st_read(file.path(folder, '..', 'data', 'raw', 'shapefiles',
                             'gadm41_UGA_2.shp'))

mri_data_uga <- read.csv(file.path(folder, '..', 'results', 'processed',
                                   'UGA_dhs',
                                   'file7_UGA_malaria_monthly_risk_covariates.csv'))

# Annual mean MRI per cluster location. Each location is surveyed in one
# year only, so it has no long-run mean of its own to deviate from.
annual_mri_uga <- mri_data_uga %>%
  group_by(longitude, latitude, year) %>%
  summarise(annual_mri = mean(monthly_mri, na.rm = TRUE), .groups = "drop")

# Snap cluster locations to the nearest district polygon
pts_uga <- annual_mri_uga %>%
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326)

pts_joined_uga <- st_join(pts_uga, ug_data["GID_2"], join = st_nearest_feature)

# Mean annual MRI of each district in each survey round
district_year_uga <- pts_joined_uga %>%
  st_drop_geometry() %>%
  group_by(GID_2, year) %>%
  summarise(annual_mri  = mean(annual_mri, na.rm = TRUE),
            n_locations = n(),
            .groups = "drop")

# Long-run mean of each district across the survey rounds in which it was
# sampled, and the anomaly from it. A district sampled in one round only has
# nothing to deviate from, so its anomaly is left missing.
district_anomaly_uga <- district_year_uga %>%
  group_by(GID_2) %>%
  mutate(n_rounds      = n(),
         location_mean = mean(annual_mri, na.rm = TRUE),
         mri_anomaly   = ifelse(n_rounds >= 2, annual_mri - location_mean,
                                NA_real_)) %>%
  ungroup()

message(sprintf("%d of %d districts were sampled in two or more survey rounds",
                n_distinct(district_anomaly_uga$GID_2[district_anomaly_uga$n_rounds >= 2]),
                n_distinct(district_anomaly_uga$GID_2)))

# Join to shapefile — filter NA year before AND after join
combined_uga <- ug_data %>%
  left_join(district_anomaly_uga %>% filter(!is.na(year)), by = "GID_2") %>%
  filter(!is.na(year))

combined_uga <- combined_uga %>%
  filter(year >= 2007 & year <= 2024)

mri_range_uga <- range(district_anomaly_uga$mri_anomaly, na.rm = TRUE)

annual_mri_plot_uga <- ggplot() +
  geom_sf(data = national_shp_uga, fill = "grey85", color = "black", linewidth = 0.3) +
  geom_sf(data = combined_uga, aes(fill = mri_anomaly), color = NA) +
  scale_fill_viridis_c(option = "viridis", limits = mri_range_uga,
                       na.value = "grey85", name = "MRI anomaly") +
  facet_wrap(~ year, nrow = 1, drop = TRUE,
             labeller = labeller(year = function(x) as.integer(x))) +
  labs(title = "(A) Uganda: 2009, 2014, 2018 and 2024",
       subtitle = "Difference between each district's mean annual MRI in a survey round and its mean over the survey rounds in which it was sampled.") +
  theme_minimal() +
  theme(legend.position = 'bottom',
        plot.margin = margin(0, 0, 0, 0),              
        plot.title = element_text(size = 9, face = "bold"),
        plot.subtitle = element_text(size = 8),
        plot.caption = element_text(size = 5, hjust = 0),
        axis.title.y = element_text(size = 7),
        axis.title.x = element_text(size = 7),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        strip.text = element_text(size = 6),
        axis.text.x = element_text(size = 5.5),
        axis.text.y = element_text(size = 6.5),
        axis.line.x  = element_line(size = 0.15),
        axis.line.y  = element_line(size = 0.15),
        legend.title = element_text(size = 6),
        legend.text = element_text(size = 5))


##################
## ZWE MRI MAPS ##
##################
national_shp_zmb = st_read(file.path(folder, '..', 'data', 'raw', 'shapefiles', 
                                 'gadm41_ZMB_0.shp'))
zmb_data <- st_read(file.path(folder, '..', 'data', 'raw', 'shapefiles', 
                             'gadm41_ZMB_2.shp'))
mri_data_zmb <- read.csv(file.path(folder, '..', 'results', 'processed',
                                   'ZMB_dhs',
                                   'file7_ZMB_malaria_monthly_risk_covariates.csv'))
# Annual mean MRI per cluster location. Each location is surveyed in one
# year only, so it has no long-run mean of its own to deviate from.
annual_mri_zmb <- mri_data_zmb %>%
  group_by(longitude, latitude, year) %>%
  summarise(annual_mri = mean(monthly_mri, na.rm = TRUE), .groups = "drop")

# Snap cluster locations to the nearest district polygon
pts_zmb <- annual_mri_zmb %>%
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326)

pts_joined_zmb <- st_join(pts_zmb, zmb_data["GID_2"], join = st_nearest_feature)

# Mean annual MRI of each district in each survey round
district_year_zmb <- pts_joined_zmb %>%
  st_drop_geometry() %>%
  group_by(GID_2, year) %>%
  summarise(annual_mri  = mean(annual_mri, na.rm = TRUE),
            n_locations = n(),
            .groups = "drop")

# Long-run mean of each district across the survey rounds in which it was
# sampled, and the anomaly from it. A district sampled in one round only has
# nothing to deviate from, so its anomaly is left missing.
district_anomaly_zmb <- district_year_zmb %>%
  group_by(GID_2) %>%
  mutate(n_rounds      = n(),
         location_mean = mean(annual_mri, na.rm = TRUE),
         mri_anomaly   = ifelse(n_rounds >= 2, annual_mri - location_mean,
                                NA_real_)) %>%
  ungroup()

message(sprintf("%d of %d districts were sampled in two or more survey rounds",
                n_distinct(district_anomaly_zmb$GID_2[district_anomaly_zmb$n_rounds >= 2]),
                n_distinct(district_anomaly_zmb$GID_2)))

# Join to shapefile — filter NA year before AND after join
combined_zmb <- zmb_data %>%
  left_join(district_anomaly_zmb %>% filter(!is.na(year)), by = "GID_2") %>%
  filter(!is.na(year))

combined_zmb <- combined_zmb %>%
  filter(year >= 2007 & year <= 2024)

mri_range_zmb <- range(district_anomaly_zmb$mri_anomaly, na.rm = TRUE)

annual_mri_plot_zmb <- ggplot() +
  geom_sf(data = national_shp_zmb, fill = "grey85", color = "black", linewidth = 0.3) +
  geom_sf(data = combined_zmb, aes(fill = mri_anomaly), color = NA) +
  scale_fill_viridis_c(option = "viridis", limits = mri_range_uga,
                       na.value = "grey85", name = "MRI anomaly") +
  facet_wrap(~ year, nrow = 1, drop = TRUE,
             labeller = labeller(year = function(x) as.integer(x))) +
  labs(title = "(B) Zambia: 2007, 2013, 2018 and 2024",
       subtitle = "Difference between each district's mean annual MRI in a survey round and its mean over the survey rounds in which it was sampled.",
       caption = "Grey: district sampled in one survey round only or no data at all.") +
  theme_minimal() +
  theme(legend.position = 'bottom',
        plot.margin = margin(0, 0, 0, 0),              
        plot.title = element_text(size = 9, face = "bold"),
        plot.subtitle = element_text(size = 8),
        plot.caption = element_text(size = 8, hjust = 0),
        axis.title.y = element_text(size = 7),
        axis.title.x = element_text(size = 7),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        strip.text = element_text(size = 6),
        axis.text.x = element_text(size = 5.5),
        axis.text.y = element_text(size = 6.5),
        axis.line.x  = element_line(size = 0.15),
        axis.line.y  = element_line(size = 0.15),
        legend.title = element_text(size = 6),
        legend.text = element_text(size = 5))

combined_plots <- ggarrange(annual_mri_plot_uga, annual_mri_plot_zmb,
                            ncol = 1, nrow = 2, align = 'hv')


path = file.path(folder, 'figures', 'pp2_fig2_monthly_mri_anomaly_plot.png')
png(path, units="in", width=8, height=6.5, res=480)
print(combined_plots)
dev.off()


anim_plot <- ggplot() +
  geom_sf(data = national_shp,
          fill = NA, color = "black", linewidth = 0.4) +
  geom_sf(data = combined,
          aes(fill = mri_anomaly), color = NA) +
  scale_fill_viridis_c(
    option   = "viridis",
    limits   = mri_range,
    na.value = "grey85",
    name     = "MRI anomaly"
  ) +
  labs(
    title    = "Annual MRI Anomaly — Zimbabwe",
    subtitle = "Year: {closest_state}",
    caption  = "Deviation of district mean MRI from long-run mean across 2015-2024"
  ) +
  theme_minimal() +
  theme(
    legend.position  = "bottom",
    legend.key.width = unit(1.8, "cm"),
    plot.title       = element_text(size = 14, face = "bold"),
    plot.subtitle    = element_text(size = 13, face = "bold", color = "black"),
    plot.caption     = element_text(size = 7,  color = "grey50"),
    axis.text.x      = element_text(size = 6),
    axis.text.y      = element_text(size = 6),
    axis.title       = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border     = element_blank(),
    legend.title     = element_text(size = 8),
    legend.text      = element_text(size = 7)
  ) +
  # gganimate transition — one frame per year
  transition_states(
    states          = year,
    transition_length = 1,
    state_length    = 3
  ) +
  enter_fade() +
  exit_fade()

# ── Render and save GIF ───────────────────────────────────────────────────────
gif_path <- file.path(folder, 'figures', 'ZWE_mri_anomaly_animated.gif')

animate(
  anim_plot,
  nframes   = 100,     # total frames across all transitions
  fps       = 12,       # frames per second — lower = slower
  width     = 700,
  height    = 600,
  renderer  = gifski_renderer(gif_path)
)
