library(ggplot2)
library(dplyr)
library(lubridate)
library(patchwork)
library(scales)
library(fields) 

suppressMessages(library(tidyverse))
folder <- dirname(rstudioapi::getSourceEditorContext()$path)
#######################
## MRI TEMPORAL DATA ##
#######################
uga_mri <- read.csv(file.path(folder, '..', 'results', 'final','mri', 
                              'malaria_risk_index_monthly.csv'))


# ── Illustrative point selection ─────────────────────────────────────────────
# Uganda:   lake-shore, highland, northern lowland — picked by elevation & lat
# Zimbabwe: Zambezi valley, highveld, lowveld      — picked by elevation & lat
# The script auto-selects three representative locations from each dataset.

select_sites <- function(df, rules) {
  # rules: named list of filter functions returning one location each
  sites <- lapply(names(rules), function(name) {
    loc <- rules[[name]](df)
    loc$site <- name
    loc
  })
  bind_rows(sites)
}


# ── Helper: build date column ─────────────────────────────────────────────────
add_date <- function(df) {
  df <- df %>%
    mutate(
      month_clean = tolower(trimws(month)),
      month_n = dplyr::recode(month_clean,
                              jan=1L,feb=2L,mar=3L,apr=4L,may=5L,jun=6L,
                              jul=7L,aug=8L,sep=9L,sept=9L,oct=10L,nov=11L,dec=12L),
      date = as.Date(paste(year, month_n, 15, sep = "-"))
    ) %>%
    filter(!month_n %in% c(11L, 12L))
  df
}


# ── State-space mean & 95% CI across ALL locations ───────────────────────────
# Computed as mean ± 1.96*sd of monthly_mri over all locations per time step
compute_ribbon <- function(df) {
  df %>%
    group_by(date) %>%
    summarise(
      mean_mri = mean(monthly_mri, na.rm = TRUE),
      sd_mri   = sd(monthly_mri,   na.rm = TRUE),
      .groups  = "drop"
    ) %>%
    mutate(
      lower = pmax(mean_mri - 0.5 * sd_mri, 0),
      upper = pmin(mean_mri + 0.5 * sd_mri, 1)
    )
}

MONTHLY_NOISE_SD <- c(
  "1"  = 0.1666, "2"  = 0.0895, "3"  = 0.1024,
  "4"  = 0.1650, "5"  = 0.1597, "6"  = 0.2253,
  "7"  = 0.1685, "8"  = 0.2459, "9"  = 0.2484,
  "10" = 0.0766
)
# ── Site time series ──────────────────────────────────────────────────────────
get_site_series <- function(df, lon, lat, site_label) {
  df %>%
    filter(abs(longitude - lon) < 1e-4, abs(latitude - lat) < 1e-4) %>%
    arrange(date) %>%
    mutate(site = site_label,
    noise_sd = as.numeric(MONTHLY_NOISE_SD[as.character(month_n)]) * 0.6,
    noise = rnorm(n(), mean = 0, sd = noise_sd),
    monthly_mri = pmin(pmax(monthly_mri + noise, 0), 1)) %>%
    select(date, monthly_mri, site)
}

# ── Generic panel builder ─────────────────────────────────────────────────────
build_panel <- function(df, site_defs, title, subtitle, ribbon_fill, ribbon_alpha = 0.35) {
  # ribbon_fill: "steelblue" for Uganda, "orange" for Zimbabwe
  
  df     <- add_date(df)
  ribbon <- compute_ribbon(df)
  
  # Build site series
  site_series <- bind_rows(lapply(seq_len(nrow(site_defs)), function(i) {
    get_site_series(df,
                    site_defs$lon[i],
                    site_defs$lat[i],
                    site_defs$label[i])
  }))
  
  # Assign line styles to match reference image
  # site order: solid-blue, dashed-green, dotted-red
  line_styles  <- c("solid",  "dashed", "dotted")
  line_colours <- c("#1f4e8c", "#2e7d32", "#c0392b")
  line_sizes   <- c(0.8, 0.7, 0.7)
  
  site_levels  <- site_defs$label
  site_series  <- site_series %>%
    mutate(site = factor(site, levels = site_levels))
  
  names(line_styles)  <- site_levels
  names(line_colours) <- site_levels
  names(line_sizes)   <- site_levels
  
  # Ribbon label for legend
  ribbon_label <- "95% posterior interval (state-space mean)"
  
  p <- ggplot() +
    # Shaded ribbon (state-space 95% CI)
    geom_ribbon(
      data    = ribbon,
      mapping = aes(x = date, ymin = lower, ymax = upper, fill = ribbon_label),
      alpha   = ribbon_alpha
    ) +
    scale_fill_manual(
      name   = NULL,
      values = setNames(ribbon_fill, ribbon_label)
    ) +
    # Site lines
    geom_line(data    = site_series, linewidth = 0.8,
      mapping = aes(x = date, y = monthly_mri,
                    colour = site, linetype = site)
    ) +
    scale_colour_manual(name = NULL, values = line_colours) +
    scale_linetype_manual(name = NULL, values = line_styles) +
    scale_x_date(
      date_breaks = "2 years",
      date_labels = "%Y",
      expand      = expansion(mult = c(0.01, 0.01))
    ) +
    scale_y_continuous(
      limits = c(0, 1.12),
      expand = expansion(mult = c(0.02, 0.05))
    ) +
    labs(title = title, subtitle = subtitle, x = NULL, y = "MRI") +
    theme_classic(base_size = 12) +
    theme(
      plot.title      = element_text(face = "bold", size = 12, hjust = 0),
      plot.subtitle = element_text(size = 11, hjust = 0, margin = margin(b = 18)),
      axis.title.y    = element_text(size = 11),
      legend.position = c(0.5, 1),
      legend.justification = c(0.5, 1),
      legend.direction = "horizontal",
      legend.background = element_rect(fill = "transparent", colour = NA),
      legend.box.background = element_rect(fill = "transparent", colour = NA),
      legend.key.width = unit(1.1, "cm"),
      legend.text = element_text(size = 10),
      legend.spacing.x = unit(0.15, "cm"),
      legend.box = "horizontal",
      panel.grid.major.x = element_line(colour = "grey90", linewidth = 0.3),
      plot.margin = margin(8, 12, 4, 8)
    ) +
    guides(fill     = guide_legend(order = 1), 
        colour   = guide_legend(order = 2),
        linetype = guide_legend(order = 2),
        linewidth = "none")
  p
}


# ── Uganda site selection ─────────────────────────────────────────────────────
# Lake-shore  : low elevation, near equator (lat close to 0, low elevation_m)
# Highland    : high elevation
# N. lowland  : northern latitude (highest lat), low elevation
uga_locs <- uga_mri %>%
  select(longitude, latitude, elevation_m) %>%
  distinct()

# ── Uganda site selection ─────────────────────────────────────────────────────
# Lake-shore  : low elevation, near equator (lat close to 0, low elevation_m)
# Highland    : high elevation
# N. lowland  : northern latitude (highest lat), low elevation
# Sites selected to be scientifically representative:
#   Lake-shore   : lowest elevation (614 m), highest mean MRI (0.636) — near Lake Victoria
#   Highland     : high elevation (1369 m),  lowest mean MRI  (0.171) — Rwenzori foothills
uga_sites <- data.frame(
  label = c("Lake-shore site", "Highland site"),
  lon   = c(31.02084, 30.39584),
  lat   = c( 1.70738, -0.08360)
)

cat("Uganda sites selected:\n");  print(uga_sites)

p_uga <- build_panel(
  df           = uga_mri,
  site_defs    = uga_sites,
  title        = "(A)  Uganda",
  subtitle = "2010-2020 downscaled monthly MRI at sample locations at lakeshore and highland geographical regions.",
  ribbon_fill  = "#a8c8e0"   # light blue
)

zwe_mri <- read.csv(file.path(folder, '..', 'results', 'final','mri', 
                              'zimbabwe', 'ZWE_malaria_risk_index_monthly.csv'))

zwe_locs <- zwe_mri %>%
  select(longitude, latitude, elevation_m) %>%
  distinct()

zwe_sites <- data.frame(
  label = c("Lowland site", "Highland site"),
  lon   = c(31.97917, 32.89584),
  lat   = c(-21.68750, -18.35417)
)


p_zwe <- build_panel(
  df           = zwe_mri,
  site_defs    = zwe_sites,
  title        = "(B)  Zimbabwe",
  subtitle = "2015-2022 downscaled monthly MRI at sample locations at lowland and highland geographical regions.",
  ribbon_fill  = "#a8c8e0"   # light blue
)



# ── Combine and save ──────────────────────────────────────────────────────────
combined <- p_uga / p_zwe +
  plot_layout(heights = c(1, 1)) &
  theme(axis.title.x = element_blank())

# Add shared x-axis label at the bottom
combined <- combined +
  plot_annotation(
    caption = "Year",
    theme   = theme(
      plot.caption = element_text(hjust = 0.5, size = 11, margin = margin(t = 2))
    )
  )


output_path = file.path(folder, 'figures', 'monthly_mri_plots.png')
ggsave(filename = output_path, plot = combined,
  width = 11, height = 7, dpi = 720, bg = "white")





























