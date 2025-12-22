# Load libraries
library(readxl)
library(tidyverse)
library(ggbreak)
library(ggplot2)

file <- "U:/Users/Silvia/RutiFrishman_2025_hormones_paper/data_to_use/data_from_hormones_for_first_graph.xlsx"
output <- "U:/Users/Silvia/RutiFrishman_2025_hormones_paper/data_to_use/"
data <- read_excel(file)

var_order <- colnames(data)

# Assign color groups
color_groups <- rep("purple", length(var_order))
color_groups[1:4] <- "red"
color_groups[5:9] <- "green"

color_map <- tibble(
  Hormone = var_order,
  ColorGroup = color_groups
)

# Convert to long format for ggplot
data_long <- data %>%
  pivot_longer(cols = everything(), names_to = "Hormone", values_to = "Value") %>%
  left_join(color_map, by = "Hormone")

data_long$Hormone <- factor(data_long$Hormone, levels = var_order)

# Create boxplot with y-axis breaks
windows(width = 11, height = 10, pointsize = 11)
p <- ggplot(data_long, aes(x = Hormone, y = Value, fill = ColorGroup)) + 
  geom_boxplot(outlier.shape = 16, width = 0.6, size = 0.3, outlier.size = 1.5) +
  stat_summary(fun = "mean", geom = "point", shape = 18, size = 1.5, fill = "white", color = "black") +
  scale_fill_manual(values = c("red" = "red", "green" = "green", "purple" = "purple")) +
  scale_y_log10(
    # choose breaks that make sense for your data
    breaks = c(1, 10, 100, 1000, 10000, 1e5),
    labels = scales::label_number(accuracy = 1)
  ) +

  theme_classic() +
  theme(
   # panel.grid.major = element_line(color = "gray85", linewidth = 0.3),
  #  panel.grid.minor = element_line(color = "gray90", linewidth = 0.2),
    panel.grid.major.x = element_blank(),  # optional: remove vertical lines if not needed
   
    axis.text.x = element_text(angle = 60, hjust = 1, size = 15),
    axis.text.y = element_text(size = 12), 
    legend.position = "none",
    axis.line.y = element_line(),
    axis.ticks.length.y = unit(3, "pt"),
    axis.ticks.y = element_line(linewidth = 0.3),
    plot.margin = margin(10, 20, 10, 65),
    
    # >>> Bigger axis titles <<<
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    
    # hide the duplicated right-side axis from ggbreak
    axis.line.y.right   = element_blank(),
    axis.ticks.y.right  = element_blank(),
    axis.text.y.right   = element_blank()
  ) +
  coord_cartesian(clip = "off") +
  labs(
    y = "Concentration (pg/mg)",
    x = NULL
  )

# y positions where you want the marks
# zigzag_y <- c(3, 15, 65, 170, 2300, 9000, 20000, 28000)
# 
# # Controls for mark appearance
# x_left  <- 0.5      # position of left edge (adjust to align with y-axis)
# width   <- 0.1      # horizontal length of the zigzag
# height  <- 0.5      # vertical amplitude of the zigzag
# n_zigzags <- 2      # number of small diagonals per break
# 
# # Add diagonals for each y break
# for (y in zigzag_y) {
#   # build a zig-zag pattern
#   for (i in 0:(n_zigzags - 1)) {
#     y1 <- y - height/2 + i * height / n_zigzags
#     y2 <- y - height/2 + (i + 1) * height / n_zigzags
#     p <- p +
#       annotate("segment",
#                x = x_left,     y = y1,
#                xend = x_left + width, yend = y2,
#                linewidth = 0.6, colour = "black")
#   }
# }
# 
# # Make sure nothing is clipped
# p <- p + coord_cartesian(clip = "off")
print(p)

ggsave(paste(output,"allbiomarkers_log_scale_vs2.pdf", sep=""), device="pdf")

