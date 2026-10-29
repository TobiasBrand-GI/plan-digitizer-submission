library(ggplot2)
library(dplyr)
library(tidyr)

res <- read.csv("data/spatial_relation_accuracy.csv", sep = ";")

overall_relate <- res %>%
  group_by(method) %>%
  summarise(n_ok = sum(ok), n = n(), .groups = "drop") %>%
  rowwise() %>%
  mutate(acc   = n_ok / n) %>%
  ungroup()

method_cols <- c("This Pipeline" = "#1b6ca8", "Opus 5.5" = "#d1495b")
zero_line   <- geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.15)


# (a) --- Map trend of accuracy against the number of areas per plan

acc_relate_areas <- res %>%
  group_by(method, plan_amount) %>%
  summarise(acc = mean(ok), n = n(), .groups = "drop") %>%
  mutate(method = factor(method, levels = c("This Pipeline", "Opus 5.5")))

p_relate_areas <- ggplot(acc_relate_areas, aes(plan_amount, acc, colour = method)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.6) +
  scale_x_continuous(breaks = function(x) seq(ceiling(x[1]), floor(x[2]), by = 1)) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  scale_colour_manual(values = method_cols, name = NULL) +
  labs(title = "Relation Accuracy vs. #Areas",
       x = "Number of areas per plan", y = "Accuracy") +
  theme_minimal(base_size = 8) +
  theme(panel.grid.minor = element_blank(),
        plot.title       = element_text(face = "bold", size = 8),
        legend.position  = "bottom",
        plot.background  = element_rect(fill = "white", colour = NA))

ggsave("corr_accuracy_size_area_relation.pdf", p_relate_areas, width = 7, height = 5, units = "cm")

# (b) --- Map trend of accuracy against the number of text blocks per plan

acc_relate_text <- res %>%
  group_by(method, text_amount) %>%
  summarise(acc = mean(ok), n = n(), .groups = "drop") %>%
  mutate(method = factor(method, levels = c("This Pipeline", "Opus 5.5")))

p_relate_text <- ggplot(acc_relate_text, aes(text_amount, acc, colour = method)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.6) +
  scale_x_continuous(breaks = function(x) seq(ceiling(x[1]), floor(x[2]), by = 1)) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  scale_colour_manual(values = method_cols, name = NULL) +
  labs(title = "Relation Accuracy vs. #Text Blocks",
       x = "Number of text blocks per plan", y = "Accuracy") +
  theme_minimal(base_size = 8) +
  theme(panel.grid.minor = element_blank(),
        plot.title       = element_text(face = "bold", size = 8),
        legend.position  = "bottom",
        plot.background  = element_rect(fill = "white", colour = NA))

ggsave("corr_accuracy_size_text_relation.pdf", p_relate_text, width = 7, height = 5, units = "cm")