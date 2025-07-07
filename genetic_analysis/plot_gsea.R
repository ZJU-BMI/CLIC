library(tidyverse)
library(ggplot2)
library(R.utils)

magma_data <- read.table("results/magma_top20.txt", header = TRUE, sep = "\t")
# top 10
# magma_data <- magma_data[1:20, ]

magma_data$log10P <- -log10(magma_data$P)
# geneset type GOBP, GOMF, REACTOME, LU
magma_data$set_type <- ifelse(grepl("GOBP_", magma_data$FULL_NAME), "GO Biological Process",
                          ifelse(grepl("GOMF_", magma_data$FULL_NAME), "GO Molecular Function",
                                 ifelse(grepl("REACTOME_", magma_data$FULL_NAME), "Reactome Pathway", "Other")))
magma_data$Simplified_Name <- gsub("GOBP_|GOMF_|REACTOME_|LU_|_DN|_UP|ABDULRAHMAN_", "", magma_data$FULL_NAME) 
magma_data$Simplified_Name <- gsub("_", " ", magma_data$Simplified_Name)
# Simplified_Name to upper case in the first letter
magma_data$Simplified_Name <- capitalize(tolower(magma_data$Simplified_Name))


# sort by log10P
magma_data_sorted <- magma_data %>%
  arrange(desc(log10P)) %>% mutate(Simplified_Name = factor(Simplified_Name, levels = Simplified_Name))

# dotplot
p <- ggplot(magma_data_sorted, aes(x = BETA, y = Simplified_Name)) +
  geom_point(aes(size = NGENES, color = P), alpha = 0.8) +
  scale_color_gradient(low = "red", high = "blue", name = "P-value",
                       trans = "log10", breaks = c(1e-8, 1e-7, 1e-6, 1e-5, 1e-4), 
                       labels = c("1e-8", "1e-7", "1e-6", "1e-5", "1e-4")) +
  scale_size_continuous(range = c(4, 12), name = "Gene num.") + 
  labs(title = "",
       x = "Rich Factor", # 
       y = "") + # 
  # xlim 0-3
  xlim(0, 3) +
  theme_bw() + 
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text.y = element_text(size = 12),
    axis.text.x = element_text(size = 13),
    axis.title.x = element_text(size = 14, face = 'bold'),
    axis.title.y = element_text(size = 14), 
    legend.position = "right",
    legend.title = element_text(size = 11),
    legend.text = element_text(size = 10),
    
    # panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 2)
  )
p
ggsave("results/fuma/magma_top20_dotplot.pdf", width = 14, height = 8, dpi = 300)
