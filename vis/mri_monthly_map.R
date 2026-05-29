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

mri_data_uga <- read.csv(file.path(folder, '..', 'results', 'final',
                                   'mri',
                                   'malaria_risk_index.csv'))
# Annual mean MRI per grid location
annual_mri_uga <- mri_data_uga %>%
  group_by(longitude, latitude, year) %>%
  summarise(annual_mri = mean(mri_value, na.rm = TRUE), .groups = "drop")

# Long-run location mean across all years
location_mean_uga <- annual_mri_uga %>%
  group_by(longitude, latitude) %>%
  summarise(location_mean = mean(annual_mri, na.rm = TRUE), .groups = "drop")

# Compute anomaly
annual_anomaly_uga <- annual_mri_uga %>%
  left_join(location_mean_uga, by = c("longitude", "latitude")) %>%
  mutate(mri_anomaly = annual_mri - location_mean)

# Snap grid points to nearest district polygon
pts_uga <- annual_anomaly_uga %>%
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326)

pts_joined_uga <- st_join(pts_uga, ug_data["GID_2"], join = st_nearest_feature)

district_anomaly_uga <- pts_joined_uga %>%
  st_drop_geometry() %>%
  group_by(GID_2, year) %>%
  summarise(
    mri_anomaly   = mean(mri_anomaly,   na.rm = TRUE),
    annual_mri    = mean(annual_mri,    na.rm = TRUE),
    location_mean = mean(location_mean, na.rm = TRUE),
    .groups = "drop"
  )

# Join to shapefile — filter NA year before AND after join
combined_uga <- ug_data %>%
  left_join(district_anomaly_uga %>% filter(!is.na(year)), by = "GID_2") %>%
  filter(!is.na(year))

combined_uga <- combined_uga %>%
  filter(year >= 2009 & year <= 2020)

mri_range_uga <- range(district_anomaly_uga$mri_anomaly, na.rm = TRUE)

annual_mri_plot_uga <- ggplot() +
  geom_sf(data = national_shp_uga, fill = NA, color = "black", linewidth = 0.3) +
  geom_sf(data = combined_uga, aes(fill = mri_anomaly), color = NA) +
  scale_fill_viridis_c(option = "viridis", limits = mri_range_uga,
    na.value = "grey85", name = "MRI anomaly") +
  facet_wrap(~ year, nrow = 3, drop = TRUE,
             labeller = labeller(year = function(x) as.integer(x))) +
  labs(title = "Uganda (2009–2020).",
    subtitle = "Deviation of mean annual MRI anomaly from country's long-run mean across all years."
  ) +
  theme_minimal() +
  theme(legend.position = 'bottom',
        plot.margin = margin(0, 0, 0, 0),              
        plot.title = element_text(size = 9, face = "bold"),
        plot.subtitle = element_text(size = 7),
        axis.title.y = element_text(size = 7),
        axis.title.x = element_text(size = 7),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        strip.text = element_text(size = 5),
        axis.text.x = element_text(size = 5),
        axis.text.y = element_text(size = 5),
        axis.line.x  = element_line(size = 0.15),
        axis.line.y  = element_line(size = 0.15),
        legend.title = element_text(size = 6),
        legend.text = element_text(size = 5)) 

path = file.path(folder, 'figures', 'UGA_mri_anomaly_plot.png')
png(path, units="in", width=7, height=6, res=300)
print(annual_mri_plot_uga)
dev.off()

##################
## ZWE MRI MAPS ##
##################
national_shp = st_read(file.path(folder, '..', 'data', 'raw', 'shapefiles', 
                                 'gadm41_ZWE_0.shp'))
zwe_data <- st_read(file.path(folder, '..', 'data', 'raw', 'shapefiles', 
                             'gadm41_ZWE_2.shp'))
mri_data <- read.csv(file.path(folder, '..', 'results', 'final',
                               'mri', 'zimbabwe',
                               'ZWE_malaria_risk_index_yearly.csv'))
annual_mri <- mri_data %>%
  group_by(longitude, latitude, year) %>%
  summarise(annual_mri = mean(mri_value, na.rm = TRUE), .groups = "drop")

# compute long-run location mean (across all years)
location_mean <- annual_mri %>%
  group_by(longitude, latitude) %>%
  summarise(location_mean = mean(annual_mri, na.rm = TRUE), .groups = "drop")

# join and compute anomaly 
annual_anomaly <- annual_mri %>%
  left_join(location_mean, by = c("longitude", "latitude")) %>%
  mutate(mri_anomaly = annual_mri - location_mean)

# snap grid points to nearest district polygon 
pts <- annual_anomaly %>%
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326)
pts_joined <- st_join(pts, zwe_data["GID_2"], join = st_nearest_feature)

district_anomaly <- pts_joined %>%
  st_drop_geometry() %>%
  group_by(GID_2, year) %>%
  summarise(
    mri_anomaly    = mean(mri_anomaly,  na.rm = TRUE),
    annual_mri     = mean(annual_mri,   na.rm = TRUE),
    location_mean  = mean(location_mean, na.rm = TRUE),
    .groups = "drop"
  )

combined <- zwe_data %>%
  left_join(district_anomaly %>% filter(!is.na(year)), by = "GID_2") %>%
  filter(!is.na(year))

mri_range <- range(district_anomaly$mri_anomaly, na.rm = TRUE)

annual_mri_plot_zwe <- ggplot() + 
  geom_sf(data = national_shp, fill = NA, color = "black", linewidth = 0.3) +
  geom_sf(data = combined, aes(fill = mri_anomaly), color = NA) +
  scale_fill_viridis_c(option = "viridis", limits   = mri_range,
    na.value = "grey85", name = "MRI anomaly") +
  facet_wrap(~ year, nrow = 2, drop = TRUE) +
  labs(title    = "Zimbabwe (2015–2022).",
    subtitle = "Deviation of mean annual MRI anomaly from country's long-run mean across all years.") +
  theme_minimal() +
  theme(legend.position = 'bottom',
        plot.margin = margin(0, 0, 0, 0),              
        plot.title = element_text(size = 9, face = "bold"),
        plot.subtitle = element_text(size = 7),
        axis.title.y = element_text(size = 7),
        axis.title.x = element_text(size = 7),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        strip.text = element_text(size = 5),
        axis.text.x = element_text(size = 5),
        axis.text.y = element_text(size = 5),
        axis.line.x  = element_line(size = 0.15),
        axis.line.y  = element_line(size = 0.15),
        legend.title = element_text(size = 6),
        legend.text = element_text(size = 5)) 


path = file.path(folder, 'figures', 'ZWE_mri_anomaly_plot.png')
png(path, units="in", width=6, height=4, res=300)
print(annual_mri_plot_zwe)
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
