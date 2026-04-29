library(ggplot2)
library(dplyr)
library(sf)
library(gganimate)
library(gifski)
library(scales)

folder <- dirname(rstudioapi::getSourceEditorContext()$path)

# ── CONFIG ────────────────────────────────────────────────────────────────────
nation_shp_path <- file.path(folder, '..', 'data', 'raw', 'shapefiles',
                             'gadm41_UGA_0.shp')
region_shp_path <- file.path(folder, '..', 'data', 'raw', 'shapefiles',
                             'gadm41_UGA_2.shp')
mri_csv_path    <- file.path(folder, '..', 'results', 'final',
                             'region_2_year_mri.csv')
output_gif      <- file.path(folder, 'figures', 'UGA_mri_timelapse.gif')

YEARS      <- 2000:2020
FPS        <- 8          # frames per second (lower = slower, higher = faster)
PAUSE_END  <- 8          # extra pause on final frame (in frames)
WIDTH_PX   <- 800
HEIGHT_PX  <- 700
# ─────────────────────────────────────────────────────────────────────────────

# ── Load data ─────────────────────────────────────────────────────────────────
cat("Loading shapefiles...\n")
nation_shp <- st_read(nation_shp_path, quiet = TRUE)
zw_data    <- st_read(region_shp_path, quiet = TRUE)

cat("Loading MRI data...\n")
zw_mri_data <- read.csv(mri_csv_path)

# Join and filter
combined <- zw_data %>%
  left_join(zw_mri_data, by = "GID_2") %>%
  filter(!is.na(year) & year %in% YEARS)

cat(sprintf("  %d region-year records across years: %s\n",
            nrow(combined), paste(sort(unique(combined$year)), collapse = ", ")))

# ── Build animated plot ───────────────────────────────────────────────────────
cat("Building animated plot...\n")

p <- ggplot() +
  geom_sf(data    = nation_shp,
          fill    = NA,
          colour  = "black",
          linewidth = 0.3) +
  geom_sf(data    = combined,
          mapping = aes(fill = mean_mri),
          colour  = NA) +
  scale_fill_viridis_c(
    option   = "viridis",
    na.value = "grey90",
    limits   = c(0.20, 0.97),
    oob      = scales::squish,
    trans    = "sqrt",
    breaks   = c(0.2, 0.4, 0.6, 0.75, 0.85, 0.95),
    labels   = c("0.2", "0.4", "0.6", "0.75", "0.85", "0.95")
  ) +
  labs(
    title    = "Malaria Risk Index (MRI) — Uganda",
    subtitle = "Year: {closest_state}",
    fill     = "MRI Value",
    caption  = "MRI derived via PCA from net access, net use, parasite, incidence and mortality rates."
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title       = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle    = element_text(size = 13, hjust = 0.5,
                                   margin = margin(b = 6)),
    plot.caption     = element_text(size = 7, colour = "grey50",
                                   hjust = 0.5),
    legend.position  = "bottom",
    legend.title     = element_text(size = 9),
    legend.text      = element_text(size = 8),
    legend.key.width = unit(2.5, "cm"),
    legend.key.height = unit(0.4, "cm"),
    axis.text        = element_text(size = 7),
    panel.grid.major = element_line(colour = "grey92", linewidth = 0.3),
    plot.margin      = margin(10, 10, 10, 10)
  ) +
  # gganimate: transition through years
  transition_states(
    states          = year,
    transition_length = 0.01,   # time spent transitioning between frames
    state_length    = 0.01      # time spent on each frame
  ) +
  ease_aes("linear")

# ── Render GIF ────────────────────────────────────────────────────────────────
cat("Rendering GIF (this may take a minute)...\n")

n_frames <- length(YEARS) * 10 + PAUSE_END   # ~10 frames per year + pause

anim <- animate(
  p,
  nframes   = n_frames,
  fps       = FPS,
  width     = WIDTH_PX,
  height    = HEIGHT_PX,
  renderer  = gifski_renderer(output_gif),
  end_pause = PAUSE_END
)

cat(sprintf("\n✓ GIF saved → %s\n", output_gif))
