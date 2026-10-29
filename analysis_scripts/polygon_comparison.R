# Polygon comparison plots: generated polygons vs. manual ground truth
# Input CSV columns:
#   id, method, area, area_partner, perimeter_gt, perimeter_pred, iou, x, y
#   (areas in m², perimeters in m, x/y = centroid coordinates for the map)
#
# Packages: install.packages(c("ggplot2", "dplyr", "patchwork", "scales"))

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(scales)

# ---- 1. Load data and compute metrics --------------------------------------
df <- read.csv("data/ground_truth_comparison_envelopes.csv")
#df <- read.csv("data/ground_truth_comparison_areas.csv")

#polsby_popper <- function(area, perimeter) 4 * pi * area / perimeter^2

df <- df %>%
  mutate(
    dA_pct   = (area_partner - area) / area * 100,     # signed relative area difference
    logratio = log2(area_partner / area),                  # symmetric alternative
    C_gt     = pp,
    C_pred   = pp_partner,
    dC       = C_pred - C_gt                               # absolute compactness difference
  )


# Summary table (median + IQR, better than mean for skewed errors)
df %>%
  group_by(method) %>%
  summarise(
    n         = n(),
    median_dA = median(dA_pct),
    IQR_dA    = IQR(dA_pct),
    median_dC = median(dC),
    median_IoU = median(iou)
  ) %>%
  print()

# ---- 2. Shared style --------------------------------------------------------
theme_set(
  theme_minimal(base_size = 11) +
    theme(panel.grid.minor = element_blank(),
          legend.position  = "bottom",
          plot.title       = element_text(face = "bold"))
)
method_cols <- c("This Pipeline" = "#1b6ca8", "Opus 5.5" = "#d1495b")
zero_line   <- geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.15)

df <- df %>%
  mutate(method = factor(method, levels = names(method_cols)))

###############################################################################
# ---- 3. Plots ---------------------------------------------------------------
###############################################################################

# (a) Distribution of area differences: violin + box
p_dist <- ggplot(df, aes(method, dA_pct, fill = method)) +
  zero_line +
  geom_violin(alpha = 0.35, colour = NA, width = 0.9) +
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.9, linewidth = 0.15) +
  scale_fill_manual(values = method_cols, guide = "none") +
  labs(title = "Area difference (IoU)", x = NULL, y = "Area Diff. [%]") +
  theme_minimal(base_size = 8) +
  theme(panel.grid.minor   = element_blank(),
        panel.grid.major.x = element_blank(),
        plot.title         = element_text(face = "bold", size = 8,
                                          margin = margin(b = 2)),
        axis.title.y       = element_text(margin = margin(r = 2)),
        plot.margin        = margin(2, 4, 2, 2))

ggsave("dist_iouDiff_areas.pdf", p_dist, width = 7, height = 5, units = "cm")

# (b) Distribution of compactness differences: violin + box
p_dist_pp <- ggplot(df, aes(method, dC, fill = method)) +
  zero_line +
  geom_violin(alpha = 0.35, colour = NA, width = 0.9) +
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.9, linewidth = 0.15) +
  scale_fill_manual(values = method_cols, guide = "none") +
  labs(title = "Difference in Compactness", x = NULL, y = "Abs. Diff. Compactness")+
  theme_minimal(base_size = 8) +
  theme(panel.grid.minor   = element_blank(),
        panel.grid.major.x = element_blank(),
        plot.title         = element_text(face = "bold", size = 8,
                                          margin = margin(b = 2)),
        axis.title.y       = element_text(margin = margin(r = 2)),
        plot.margin        = margin(2, 4, 2, 2))

ggsave("dist_compactDiff_areas.pdf", p_dist_pp, width = 7, height = 5, units = "cm")


# (c) Combined error space: area vs. compactness, coloured by IoU
p_space <- ggplot(df, aes(dA_pct, dC, colour = method)) +
  geom_hline(yintercept = 0, colour = "grey60") +
  geom_vline(xintercept = 0, colour = "grey60") +
  geom_point(size = 1.6, alpha = 0.85) +
  geom_smooth(method = MASS::rlm, formula = y ~ x, se = FALSE, linewidth = 0.6) +
  annotate("text", x = Inf,  y = Inf,  label = "larger & more compact",
           hjust = 1.05, vjust = 1.5, size = 2, colour = "grey40") +
  annotate("text", x = -Inf, y = -Inf, label = "smaller & less compact",
           hjust = -0.05, vjust = -0.8, size = 2, colour = "grey40") +
  scale_colour_manual(values = method_cols, name = NULL) +
  facet_wrap(~ method) +
  labs(title = "Combined error space", x = "Area Diff. [%]", y = "Abs. Diff. Compactness") +
  theme_minimal(base_size = 8) +
  theme(panel.grid.minor   = element_blank(),
        panel.grid.major.x = element_blank(),
        plot.title         = element_text(face = "bold", size = 8,
                                          margin = margin(b = 2)),
        axis.title.y       = element_text(margin = margin(r = 2)),
        plot.margin        = margin(2, 4, 2, 2))

r_lab <- df %>%
  group_by(method) %>%
  summarise(rho = cor(dA_pct, dC, method = "spearman", use = "complete.obs"),
            p   = cor.test(dA_pct, dC, method = "spearman", exact = FALSE)$p.value) %>%
  mutate(label = ifelse(
    p < 0.001,
    sprintf("rho == '%.2f' * ',' ~ italic(p) < '.001'", rho),
    sprintf("rho == '%.2f' * ',' ~ italic(p) == '%.3f'", rho, p)
  ))

p_space <- p_space +
  geom_text(data = r_lab, aes(x = -Inf, y = -Inf, label = label),
            parse = TRUE,
            hjust = -0.05, vjust = -2.3, size = 3, colour = "grey20",
            inherit.aes = FALSE)

ggsave("comb_errorspace_areas.pdf", p_space, width = 14, height = 10, units = "cm")


# (d) Correlation between errors and area size
df_long <- df %>%
  pivot_longer(c(dA_pct, dC), names_to = "measure", values_to = "error") %>%
  mutate(measure = factor(measure, levels = c("dA_pct", "dC"),
                          labels = c("Area Diff. [%]",
                                     "Abs. Diff. Compactness")))

rho_lab <- df_long %>%
  group_by(measure, method) %>%
  summarise(rho = cor(area, error, method = "spearman", use = "complete.obs"),
            p   = cor.test(area, error, method = "spearman", exact = FALSE)$p.value,
            .groups = "drop") %>%
  mutate(label = ifelse(
    p < 0.001,
    sprintf("rho == '%.2f' * ',' ~ italic(p) < '.001'", rho),
    sprintf("rho == '%.2f' * ',' ~ italic(p) == '%.3f'", rho, p)
  ),
  vjust = ifelse(method == unique(method)[1], -2, -0.5))   # stack labels for the two methods

p_size <- ggplot(df_long, aes(area, error, colour = method)) +
  geom_hline(yintercept = 0, colour = "grey60") +
  geom_point(size = 1.4, alpha = 0.7) +
  geom_smooth(method = MASS::rlm, formula = y ~ x, se = FALSE, linewidth = 0.6) +
  geom_text(data = rho_lab,
            aes(x = Inf, y = -Inf, label = label, vjust = vjust),
            parse = TRUE, hjust = 1.05, size = 2.8, show.legend = FALSE) +
  facet_wrap(~ measure, scales = "free_y", strip.position = "left") +
  scale_colour_manual(values = method_cols, name = NULL) +
  labs(title = "Error vs. object size", x = "Ground-truth area [m²]", y = NULL) +
  theme_minimal(base_size = 8) +
  theme(panel.grid.minor  = element_blank(),
        plot.title        = element_text(face = "bold", size = 8),
        legend.position   = "bottom",
        strip.placement   = "outside",
        strip.background  = element_blank(),
        strip.text.y.left = element_text(size = 8, angle = 90,
                                         margin = margin(r = 2)))

ggsave("corr_errors_size_env.pdf", p_size, width = 14, height = 10, units = "cm")

# (e) Correlation between errors and amount of items per plan
items_long <- df %>%
  mutate(abs_dA = abs(dA_pct),
         abs_dC = abs(dC)) %>%
  pivot_longer(c(abs_dA, abs_dC), names_to = "measure", values_to = "error") %>%
  mutate(measure = recode(measure,
                          abs_dA = "Mean abs. area diff. [%]",
                          abs_dC = "Mean abs. diff. compactness"))
items_long <- items_long %>%
  mutate(measure = factor(measure,
                          levels = c("Mean abs. area diff. [%]",
                                     "Mean abs. diff. compactness")))

pd <- position_dodge(width = 0.4)

p_items <- ggplot(items_long, aes(plan_amount, error, colour = method)) +
  stat_summary(fun = mean, geom = "line", position = pd, linewidth = 0.6) +
  stat_summary(fun.data = mean_se, geom = "pointrange", position = pd, size = 0.3) +
  facet_wrap(~ measure, scales = "free_y", strip.position = "left") +
  scale_x_continuous(breaks = function(x) seq(ceiling(x[1]), floor(x[2]), by = 1)) +
  scale_colour_manual(values = method_cols, name = NULL) +
  labs(title = "Error vs. number of items per plan",
       x = "Number of items per plan", y = NULL) +
  theme_minimal(base_size = 8) +
  theme(panel.grid.minor  = element_blank(),
        plot.title        = element_text(face = "bold", size = 8),
        legend.position   = "bottom",
        strip.placement   = "outside",
        strip.background  = element_blank(),
        strip.text.y.left = element_text(size = 8, angle = 90, margin = margin(r = 2)))

ggsave("corr_errors_amount_areas.pdf", p_items, width = 16, height = 8, units = "cm")

###############################################################################
# ---- 4. Values ---------------------------------------------------------------
###############################################################################

#df_raw <- read.csv("data/ground_truth_comparison_envelopes.csv")
df_raw <- read.csv("data/ground_truth_comparison_areas.csv")

#polsby_popper <- function(area, perimeter) 4 * pi * area / perimeter^2

df_raw <- df_raw %>%
  mutate(
    dA_pct   = (area_partner - area) / area * 100,     # signed relative area difference
    logratio = log2(area_partner / area),                  # symmetric alternative
    C_gt     = pp,
    C_pred   = pp_partner,
    dC       = C_pred - C_gt                               # absolute compactness difference
  )


vals_iou <- df_raw %>%
  group_by(method) %>%
  summarise(
    mean = mean(dA_pct, na.rm = TRUE),
    sd   = sd(dA_pct, na.rm = TRUE),
    n    = n()
  )

vals_compact <- df_raw %>%
  group_by(method) %>%
  summarise(
    mean = mean(dC, na.rm = TRUE),
    sd   = sd(dC, na.rm = TRUE),
    n    = n()
  )
