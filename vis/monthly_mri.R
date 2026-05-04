library(ggplot2)
library(dplyr)

zwe_mri <- read.csv(file.path(folder, '..', 'results', 'final', 'mri',
                              'zimbabwe', 'ZWE_malaria_risk_index_monthly.csv'))

df <- zwe_mri %>%
  filter(year >= 2020, year <= 2022) %>%
  mutate(month_num = as.integer(month_num)) %>%
  filter(month_num %in% 1:12)

# Compute 3-year monthly climatology and anomalies
climatology <- df %>%
  group_by(longitude, latitude, month_num) %>%
  summarise(clim_mri = mean(monthly_mri, na.rm = TRUE), .groups = "drop")

df <- df %>%
  left_join(climatology, by = c("longitude", "latitude", "month_num")) %>%
  mutate(anomaly = monthly_mri - clim_mri)

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

zwe_sites <- data.frame(
  label = c("Site 1 (159 m)", "Site 2 (2,106 m)"),
  lon   = c(32.39584,       32.77084),
  lat   = c(-21.31250,      -17.97917)
)

site_series <- bind_rows(lapply(seq_len(nrow(zwe_sites)), function(i) {
  df %>%
    filter(abs(longitude - zwe_sites$lon[i]) < 1e-4,
           abs(latitude  - zwe_sites$lat[i]) < 1e-4) %>%
    arrange(year, month_num) %>%
    mutate(site = zwe_sites$label[i]) %>%
    select(year, month_num, monthly_mri = anomaly, site)
}))

site_series$site <- factor(site_series$site, levels = zwe_sites$label)
month_labels <- c("Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec")

# ── Plot ──────────────────────────────────────────────────────────────────────
ZWE_monthly_mri <- ggplot() + geom_ribbon(data = ribbon,
    mapping = aes(x = month_num, ymin = lower, ymax = upper,
            fill = "95% posterior interval (state-space mean)"), alpha = 0.35) +
  geom_line(data = site_series, mapping = aes(x = month_num, y = monthly_mri,
                  colour = site, linetype = site), linewidth = 0.8) +
  scale_fill_manual(name = NULL, values = c("95% posterior interval (state-space mean)" = "#a8c8e0")) +
  scale_colour_manual(name = NULL, values = c("Site 1 (159 m)" = "#1f4e8c", 
                                              "Site 2 (2,106 m)" = "#2e7d32")) +
  scale_linetype_manual(name = NULL, values = c("Site 1 (159 m)" = "solid", 
                                                "Site 2 (2,106 m)" = "dashed")) +
  facet_wrap(~ year, ncol = 3, scales = "free_y") +labs(title    = "Zimbabwe Monthly MRI",
    subtitle = "2020–2022 monthly MRI anomaly relative to 3-year climatological mean for two sample sites.",
    caption  = "For each month between 2020 and 2022, we show how much the MRI at sites 1 and 2 deviated from what was typical for that month on average across those three years.",
    x = "Month", y = "MRI") + theme_minimal() +
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
    plot.caption = element_text(size = 8, hjust = 0, colour = "#2e7d32",
                                margin = margin(t = 10), face = "italic")) +
  guides(fill = guide_legend(order = 1), colour = guide_legend(order = 2),
    linetype = guide_legend(order = 2))

path = file.path(folder, 'figures', 'ZWE_monthly_mri.png')
png(path, units="in", width=10, height=4, res=300)
print(ZWE_monthly_mri)
dev.off()











