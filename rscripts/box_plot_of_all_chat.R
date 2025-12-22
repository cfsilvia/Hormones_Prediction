# Install missing packages if needed
pkgs <- c("readxl", "tidyverse", "ggbreak", "ggh4x")
to_install <- pkgs[!(pkgs %in% installed.packages()[,"Package"])]
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

library(readxl)
library(tidyverse)
library(ggbreak)
library(ggh4x)

# ---------- Load ----------
file_path <- "F:\\SilviaData\\rutiFrishman\\September2025\\October_2025\\data_from_hormones_for_first_graph.xlsx"
df <- read_excel(file_path)

# ---------- Order & Colors (no joins) ----------
var_order <- colnames(df)

# Index of each variable in the order
idx_lookup <- setNames(seq_along(var_order), var_order)

# Long format with order + color computed directly
long <- df %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Value") %>%
  mutate(
    Variable   = factor(Variable, levels = var_order),
    VarIndex   = idx_lookup[as.character(Variable)],
    ColorGroup = case_when(
      VarIndex <= 4 ~ "red",
      VarIndex <= 9 ~ "green",
      TRUE          ~ "purple"
    )
  )

# ---------- Heuristic per-variable break window ----------
# Break if the top tail (above q95) spans >= 25% of total range and q99 > q95.
break_table <- long %>%
  group_by(Variable) %>%
  summarise(
    q95  = quantile(Value, 0.95, na.rm = TRUE),
    q99  = quantile(Value, 0.99, na.rm = TRUE),
    vmax = max(Value, na.rm = TRUE),
    vmin = min(Value, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    tail_spread  = vmax - q95,
    total_spread = vmax - vmin,
    need_break   = is.finite(tail_spread) & is.finite(total_spread) &
      total_spread > 0 & (tail_spread / total_spread) >= 0.25 & (q99 > q95),
    br_low  = q95 * 0.99,   # small padding below q95
    br_high = q99 * 1.01    # small padding above q99
  )

# ---------- Build facet-specific y scales ----------
# For facets that need a break, we add scale_y_break(c(br_low, br_high)).
# (If you want MULTIPLE breaks per facet: duplicate the formula for the same Variable
# and set a second scale_y_break(c(..., ...)) with another window.)
y_scales <- list()
for (i in seq_len(nrow(break_table))) {
  v  <- break_table$Variable[i]
  nb <- break_table$need_break[i]
  if (isTRUE(nb)) {
    lo <- break_table$br_low[i]
    hi <- break_table$br_high[i]
    y_scales[[length(y_scales) + 1]] <- rlang::new_formula(
      rlang::expr(Variable == !!v),
      ggbreak::scale_y_break(c(lo, hi))
    )
  }
}

# ---------- Plot ----------
p <- ggplot(long, aes(x = Variable, y = Value, fill = ColorGroup)) +
  geom_boxplot(outlier.shape = 16, width = 0.6) +
  facet_wrap(~ Variable, scales = "free_y", ncol = 5) +
  ggh4x::facetted_pos_scales(y = y_scales) +
  scale_fill_manual(values = c(red = "red", green = "green", purple = "purple")) +
  labs(
    title = "Boxplots (Ordered, Colored) with Per-Variable Axis Breaks",
    x = NULL, y = "Value"
  ) +
  theme_bw() +
  theme(
    strip.text = element_text(face = "bold"),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    legend.position = "none",
    plot.title = element_text(hjust = 0.5)
  )
print(p)
# ---------- Save PDF ----------
#ggsave("boxplots_with_order_colors_and_breaks.pdf", plot = p, width = 16, height = 18, units = "in")

#message("Saved: boxplots_with_order_colors_and_breaks.pdf")