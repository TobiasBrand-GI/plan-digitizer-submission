library(ggplot2)
library(dplyr)
library(tidyr)

res <- read.csv("data/text_extraction_accuracy.csv", sep = ";")

# (a) --- Grouped bar plot for the accuracy per item group

acc_group <- res %>%
  group_by(method, group) %>%
  summarise(n_ok = sum(match), n = n(), .groups = "drop") %>%
  rowwise() %>%
  mutate(acc   = n_ok / n) %>%
  ungroup() %>%
  mutate(group = factor(group,
                        levels = c("Type of Use", "Extent of Construction Use", "Heights",
                                   "Roof Constraints", "Number of Dwellings")),
         group = droplevels(group))

method_cols <- c("This Pipeline" = "#1b6ca8", "Opus 5.5" = "#d1495b")
acc_group <- acc_group %>%
  mutate(method = factor(method, levels = names(method_cols)))

pd <- position_dodge(width = 0.7)

overall <- res %>%
  group_by(method) %>%
  summarise(n_ok = sum(match), n = n(), .groups = "drop") %>%
  rowwise() %>%
  mutate(acc   = n_ok / n) %>%
  ungroup()

overall <- overall %>%
  mutate(method = factor(method))

p_group <- ggplot(acc_group, aes(group, acc, fill = method)) +
  geom_col(position = pd, width = 0.65) +
  geom_hline(data = overall,
             aes(yintercept = acc),
             colour = "white",linewidth = 0.6) +
  geom_hline(data = overall,
             aes(yintercept = acc, colour = method),
             linetype = "dashed", linewidth = 0.4) +
  geom_label(aes(label = sprintf("%.0f%%\n(%d/%d)", acc * 100, n_ok, n),
                 group = method),
             position = pd, vjust = -0.3, size = 2.2, lineheight = 0.9,
             fill = "white", label.size = 0,
             label.padding = unit(0.08, "lines"), label.r = unit(0, "lines")) +
  geom_label(data = overall,
             aes(x = Inf, y = acc, colour = method,
                 label = sprintf("Ø %.0f%%", acc * 100)),
             hjust = 1.05, vjust = -0.4, size = 2.2, inherit.aes = FALSE,
             show.legend = FALSE,
             fill = "white", label.size = 0,
             label.padding = unit(0.08, "lines"), label.r = unit(0, "lines")) +
  scale_y_continuous(labels = scales::percent, breaks = seq(0, 1, 0.25),
                     limits = c(0, 1.12), expand = expansion(mult = c(0, 0.02))) +
  scale_fill_manual(values = method_cols, name = NULL) +
  scale_colour_manual(values = method_cols, guide = "none") +
  labs(title = "Match Rate per Attribute Group", x = "Attribute Group", y = "Match Rate") +
  theme_minimal(base_size = 8) +
  theme(panel.grid.minor   = element_blank(),
        panel.grid.major.x = element_blank(),
        plot.title         = element_text(face = "bold", size = 8),
        legend.position    = "bottom")

ggsave("text_accuracy_groups.pdf", p_group, width = 16, height = 8, units = "cm")


# (b) --- Map trend of accuracy against the number of items per plan

acc_n <- res %>%
  group_by(method, plan_amount) %>%
  summarise(acc = mean(match), n = n(), .groups = "drop") %>%
  mutate(method = factor(method, levels = c("This Pipeline", "Opus 5.5")))

p_text_n <- ggplot(acc_n, aes(plan_amount, acc, colour = method)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.6) +
  scale_x_continuous(breaks = function(x) seq(ceiling(x[1]), floor(x[2]), by = 1)) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  scale_colour_manual(values = method_cols, name = NULL) +
  labs(title = "Text Extraction Accuracy vs. Number of Items",
       x = "Number of items per plan", y = "Accuracy") +
  theme_minimal(base_size = 8) +
  theme(panel.grid.minor = element_blank(),
        plot.title       = element_text(face = "bold", size = 8),
        legend.position  = "bottom",
        plot.background  = element_rect(fill = "white", colour = NA))

ggsave("corr_accuracy_size_text.pdf", p_text_n, width = 16, height = 8, units = "cm")
