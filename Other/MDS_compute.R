#!/usr/bin/env Rscript

# Description:
#   This script performs Multidimensional Scaling (MDS) on various distance matrices
#   and creates visualizations of the results.
# Accepts:
#   Command line arguments:
#     1. path_to_dataframe.csv: Path to the input CSV file containing the data
#     2. path_for_npy_and_output: Path for input .npy files and output plots
# Returns:
#   None: Saves MDS plots as PNG files and displays them

# Install required packages
install.packages("reticulate", quiet = TRUE)
install.packages("patchwork", quiet = TRUE)

# Load required libraries
library(readr)
library(dplyr)
library(ggplot2)
library(tibble)
library(data.table)
library(reticulate)
library(patchwork)

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check if correct number of arguments are provided
if (length(args) != 2) {
  stop("Usage: Rscript script_name.R <path_to_dataframe.csv> <path_for_npy_and_output>")
}

# Assign command line arguments to variables
dataframe_path <- args[1]
io_path <- args[2]

# Read the input CSV file
data_sampled <- read_csv(dataframe_path)

# Function to create MDS plots
create_mds_plots <- function(distance_matrix, title_prefix) {
  # Perform Classical MDS
  mds <- cmdscale(as.dist(distance_matrix), k = 3) %>%
    as_tibble()
  
  # Add class labels to the MDS result
  mds <- mds %>%
    mutate(Class = data_sampled$Class)
  
  # Rename columns
  colnames(mds) <- c("Dim.1", "Dim.2", "Dim.3", "Class")
  
  # Create plot for Dimension 1 vs 2
  plot1 <- ggplot(mds, aes(x = Dim.1, y = Dim.2, color = Class)) +
    geom_point(size = 3, alpha = 0.35) +
    theme_minimal() +
    labs(title = paste(title_prefix, "Dim 1 vs. 2"),
         x = "MDS Dimension 1", y = "MDS Dimension 2") +
    theme(legend.title = element_text(size = 10), 
          legend.text = element_text(size = 8),
          plot.title = element_text(size = 12))
  
  # Create plot for Dimension 1 vs 3
  plot2 <- ggplot(mds, aes(x = Dim.1, y = Dim.3, color = Class)) +
    geom_point(size = 3, alpha = 0.35) +
    theme_minimal() +
    labs(title = paste(title_prefix, "Dim 1 vs. 3"),
         x = "MDS Dimension 1", y = "MDS Dimension 3") +
    theme(legend.title = element_text(size = 10), 
          legend.text = element_text(size = 8),
          plot.title = element_text(size = 12))
  
  # Create plot for Dimension 2 vs 3
  plot3 <- ggplot(mds, aes(x = Dim.2, y = Dim.3, color = Class)) +
    geom_point(size = 3, alpha = 0.35) +
    theme_minimal() +
    labs(title = paste(title_prefix, "Dim 2 vs. 3"),
         x = "MDS Dimension 2", y = "MDS Dimension 3") +
    theme(legend.title = element_text(size = 10), 
          legend.text = element_text(size = 8),
          plot.title = element_text(size = 12))
  
  # Combine plots in a tight layout
  combined_plot <- (plot1 + plot2 + plot3) +
    plot_layout(ncol = 2) +
    plot_annotation(
      title = paste(title_prefix, "MDS Dimensions"),
      theme = theme(plot.title = element_text(size = 14, hjust = 0.5))
    ) &
    theme(legend.position = "bottom")
  
  return(combined_plot)
}

# Import numpy for reading .npy files
np <- import("numpy")

# List of matrices to process
matrices <- c("correlation_matrix", "euclidean_matrix", "rbf_distances", "std_euclidean_matrix")

# Process each matrix
for (matrix in matrices) {
  # Construct full path to .npy file
  npy_path <- file.path(io_path, paste0(matrix, ".npy"))
  
  # Check if file exists
  if (!file.exists(npy_path)) {
    warning(paste("File not found:", npy_path))
    next
  }
  
  # Load matrix from .npy file
  distance_matrix <- np$load(npy_path)
  
  # Create MDS plots
  plot <- create_mds_plots(distance_matrix, tools::toTitleCase(gsub("_", " ", matrix)))
  
  # Save plot as PNG file
  output_file <- file.path(io_path, paste0(matrix, "_mds_plots.png"))
  ggsave(output_file, plot, width = 12, height = 8, dpi = 300)
  
  # Display plot
  print(plot)
}
