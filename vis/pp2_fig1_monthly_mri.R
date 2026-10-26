library(ggplot2)
library(dplyr)

suppressMessages(library(tidyverse))
folder <- dirname(rstudioapi::getSourceEditorContext()$path)

zmb_mri <- read.csv(file.path(folder, '..', 'results', 'processed',
                              'ZMB_dhs', 'file7_ZMB_malaria_monthly_risk_covariates.csv'))

df <- zmb_mri %>%
  filter(year >= 2007, year <= 2024) %>%
  mutate(month_num = as.integer(month_num)) %>%
  filter(month_num %in% 1:12)

# The file holds one row per child and month, and the children at one
# location share the same months, so take their mean. This leaves one value
# per location, year, and month.
has_elevation <- "elevation_m" %in% names(df)

df <- df %>%
  group_by(longitude, latitude, year, month_num) %>%
  summarise(monthly_mri = mean(monthly_mri, na.rm = TRUE),
            elevation_m = if (has_elevation) first(elevation_m) else NA_real_,
            .groups = "drop")

# Each location is surveyed in one year only, so it has no multi-year
# climatology. The anomaly is the deviation of each month from the mean of
# that location's own 12 months.
df <- df %>%
  group_by(longitude, latitude, year) %>%
  mutate(anomaly = monthly_mri - mean(monthly_mri, na.rm = TRUE)) %>%
  ungroup()

band_label <- "Mean \u00b1 0.5 SD across all locations"

ribbon <- df %>%
  group_by(year, month_num) %>%
  summarise(
    mean_mri = mean(anomaly, na.rm = TRUE),
    sd_mri   = sd(anomaly,   na.rm = TRUE),
    .groups  = "drop"
  ) %>%
  mutate(
    lower = mean_mri - 0.5 * sd_mri,
    upper = mean_mri + 0.5 * sd_mri
  )

# The two sites are clusters of the 2024 survey. Site 1 is the highest-risk
# cluster and Site 2 the lowest-risk one among the clusters with at least 20
# children from 12 or more households and a within-cluster risk SD below
# 0.06. The coordinates must match the data to within 1e-4 degrees.
zmb_sites <- data.frame(
  lon = c(22.3738271046, 32.7495511282),
  lat = c(-14.9981127,   -9.34533391662)
)

site_rows <- function(i) {
  df %>%
    filter(abs(longitude - zmb_sites$lon[i]) < 1e-4,
           abs(latitude  - zmb_sites$lat[i]) < 1e-4)
}

for (i in seq_len(nrow(zmb_sites))) {
  if (nrow(site_rows(i)) == 0) {
    stop(sprintf("Site %d (lon %s, lat %s) is not in the data",
                 i, zmb_sites$lon[i], zmb_sites$lat[i]))
  }
}

# A cluster is surveyed in one year only, so a site has no values in the
# other survey years. In each year the site is shown by the nearest surveyed
# cluster, which is the site itself in its own survey year.
haversine_km <- function(lon1, lat1, lon2, lat2) {
  rad <- pi / 180
  a <- sin((lat2 - lat1) * rad / 2)^2 +
    cos(lat1 * rad) * cos(lat2 * rad) * sin((lon2 - lon1) * rad / 2)^2
  2 * 6371 * asin(sqrt(a))
}

locations <- df %>% distinct(longitude, latitude, year, elevation_m)

site_locations <- bind_rows(lapply(seq_len(nrow(zmb_sites)), function(i) {
  locations %>%
    mutate(distance_km = haversine_km(longitude, latitude,
                                      zmb_sites$lon[i], zmb_sites$lat[i])) %>%
    group_by(year) %>%
    slice_min(distance_km, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    mutate(site_index = i)
}))

if (any(site_locations$distance_km > 100)) {
  message("Some plotted clusters are more than 100 km from their site, ",
          "see the distances in the caption")
}

# The label shows the elevation of the site itself when the file has it
zmb_sites$label <- sapply(seq_len(nrow(zmb_sites)), function(i) {
  elev <- site_locations %>%
    filter(site_index == i, distance_km < 0.01) %>%
    pull(elevation_m)
  if (is.na(elev[1])) {
    paste("Site", i)
  } else {
    sprintf("Site %d (%s m)", i, format(round(elev[1]), big.mark = ","))
  }
})

site_series <- df %>%
  inner_join(site_locations %>% select(site_index, longitude, latitude, year),
             by = c("longitude", "latitude", "year")) %>%
  arrange(site_index, year, month_num) %>%
  mutate(site = zmb_sites$label[site_index]) %>%
  select(year, month_num, monthly_mri = anomaly, site)

site_series$site <- factor(site_series$site, levels = zmb_sites$label)

# How far each plotted cluster is from its site, in the other years
away <- site_locations %>%
  filter(distance_km >= 0.01) %>%
  arrange(site_index, year) %>%
  group_by(site_index) %>%
  summarise(text = paste0(year, ": ", round(distance_km), " km",
                          collapse = ", "), .groups = "drop")

distance_text <- if (nrow(away) > 0) {
  paste0(" Distance from the site: ",
         paste0("Site ", away$site_index, " (", away$text, ")",
                collapse = "; "), ".")
} else {
  ""
}

caption_text <- str_wrap(paste0(
  "For each month, we show how much the MRI at sites 1 and 2 deviated from ",
  "the average of their own 12 months in the survey year. A cluster is ",
  "surveyed in one year only, so in the other years each site is shown by ",
  "the nearest surveyed cluster.", distance_text), width = 140)

month_labels <- c("Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec")

site_colours <- setNames(c("#1f4e8c", "#2e7d32"), zmb_sites$label)
site_lines   <- setNames(c("solid", "dashed"),    zmb_sites$label)

# ── Plot ──────────────────────────────────────────────────────────────────────
ZMB_monthly_mri <- ggplot() + geom_ribbon(data = ribbon,
                                          mapping = aes(x = month_num, ymin = lower, ymax = upper,
                                                        fill = band_label), alpha = 0.35) +
  geom_line(data = site_series, mapping = aes(x = month_num, y = monthly_mri,
                                              colour = site, linetype = site), linewidth = 0.8) +
  scale_fill_manual(name = NULL, values = setNames("#a8c8e0", band_label)) +
  scale_colour_manual(name = NULL, values = site_colours) +
  scale_linetype_manual(name = NULL, values = site_lines) +
  facet_wrap(~ year, ncol = 4, nrow = 1, scales = "free_y") +
  labs(title    = "Zambia Monthly MRI",
       subtitle = "Annual MRI anomaly relative to each location's own annual mean, for two sample sites.",
      caption  = caption_text, x = "Month", y = "MRI") + theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 12, hjust = 0),
        plot.subtitle = element_text(size = 11, hjust = 0, margin = margin(b = 18)),
        axis.title.y = element_text(size = 11),
        legend.position = "bottom",
        legend.justification  = c(0.5, 1),
        legend.direction = "horizontal",
        legend.background = element_rect(fill = "transparent", colour = NA),
        legend.box.background = element_rect(fill = "transparent", colour = NA),
        legend.key.width = unit(1.1, "cm"),
        legend.text = element_text(size = 8),
        legend.spacing.x = unit(0.15, "cm"),
        legend.box  = "horizontal",
        strip.text = element_text(face = "bold", size = 11),
        panel.grid.major.x = element_blank(),  
        panel.grid.major.y = element_blank(),  
        panel.grid.minor = element_blank(),   
        axis.line = element_line(colour = "black", linewidth = 0.3),  
        plot.margin = margin(8, 12, 4, 8),
        plot.caption = element_text(size = 9, hjust = 0, colour = "#2e7d32",
                                    margin = margin(t = 10), face = "italic")) +
  guides(fill = guide_legend(order = 1), colour = guide_legend(order = 2),
         linetype = guide_legend(order = 2))

path = file.path(folder, 'figures', 'pp2_fig1_ZMB_monthly_mri.png')
png(path, units="in", width=9, height=4, res=300)
print(ZMB_monthly_mri)
dev.off()











