# Stacked bar chart to visualize gene age composition of different gene groups

# load packages
library(dplyr)
library(ggplot2)
library(tidyverse)
# set working directory
setwd("path/to/your/directory")
# -----------------------------
# read your data
top_genes    <- read.csv("top500csv", sep=";", header=TRUE)
low_genes    <- read.csv("lowest500.csv", sep=";", header=TRUE)
rand_below   <- read.csv("random.csv", sep=";", header=TRUE)
rand_above   <- read.csv("random.csv", sep=";", header=TRUE)

# label groups

top_genes$Group  <- "Top"
low_genes$Group  <- "Low"
rand_below$Group <- "Random_Below"
rand_above$Group <- "Random_Above"

# merge
all_genes <- bind_rows(top_genes, low_genes, rand_below, rand_above)

# modal_value numeric
all_genes$modal_value <- as.numeric(all_genes$modal_value)

# adjust values (15.1, 15.2 -> 15, 19.1, 19.2 -> 19)
all_genes <- all_genes %>%
  mutate(modal_value = case_when(
    modal_value %in% c(19.1, 19.2) ~ 19,
    modal_value %in% c(15.1, 15.2) ~ 15,
    TRUE ~ modal_value
  ))

# remove NAs
all_genes <- all_genes %>% filter(!is.na(modal_value))

# calculate group size
group_sizes <- all_genes %>%
  group_by(Group) %>%
  summarise(n = n())

# labels for plot
group_labels <- group_sizes %>%
  mutate(label = paste0(Group, "\n(n=", n, ")"))

label_vector <- group_labels$label
names(label_vector) <- group_labels$Group
label_vector <- label_vector[c("Top", "Random_Above", "Low", "Random_Below")]

# determine order for groups
all_genes$Group <- factor(all_genes$Group,
                          levels = c("Top", "Random_Above", "Low", "Random_Below"))

# -----------------------------
# Prepare stacked distribution 


# All age values (to fill in missing bars)
all_ages <- seq(floor(min(all_genes$modal_value)),
                ceiling(max(all_genes$modal_value)), by = 1)

# Grid of all groups × all age values
full_grid <- expand.grid(
  Group = levels(all_genes$Group),
  modal_value = all_ages
)

# Calculate frequencies
freq_data <- all_genes %>%
  group_by(Group, modal_value) %>%
  summarise(count = n(), .groups = "drop")

# Fill missing combinations with count = 0
freq_data <- full_grid %>%
  left_join(freq_data, by = c("Group", "modal_value")) %>%
  mutate(count = replace_na(count, 0)) %>%
  arrange(Group, modal_value)

# Cumulative sum for stacked bars (for mean point)
freq_data <- freq_data %>%
  group_by(Group) %>%
  mutate(cum_count = cumsum(count) - count/2)

# Means per group
group_means <- all_genes %>%
  group_by(Group) %>%
  summarise(mean_value = mean(modal_value))

# Mean positions on the stacked bar
mean_positions <- freq_data %>%
  left_join(group_means, by = "Group") %>%
  group_by(Group) %>%
  summarise(y = cum_count[which.min(abs(modal_value - mean_value))],
            mean_value = unique(mean_value))


# -----------------------------
#  Stacked Distribution + Mean Marker (Legend Left)

# Calculate percentage frequencies
freq_data_percent <- freq_data %>%
  group_by(Group) %>%
  mutate(percent = count / sum(count) * 100) %>%
  ungroup()

# Prepare means as text
mean_labels <- group_means %>%
  mutate(label = paste0("Mean = ", round(mean_value,1)))
ggplot(freq_data_percent, aes(x = Group, y = percent, fill = factor(modal_value))) +
  geom_bar(stat = "identity") +  # stacked bars
  # Mean text below the bars"
  geom_text(data = mean_labels,
            aes(x = Group, y = -2, label = label),
            inherit.aes = FALSE,
            vjust = 1,
            size = 5,       # increase label size
            fontface = "bold") +
  scale_x_discrete(labels = label_vector) +
  scale_y_continuous(limits = c(-5, 100), expand = c(0,0), labels = function(x) paste0(x, "%")) +
  labs(
    x = "",
    y = "% Genes",
    fill = "Modal Value",
    title = "Distribution of Estimated Gene Age (Phylostratum)"
  ) +
  theme_minimal() +
  theme(
    legend.position = "right",
    panel.grid = element_blank(),  # remove Grid 
    axis.text = element_text(size = 12),   # bigger axis labels 
    axis.title = element_text(size = 14, face = "bold"),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
  )
