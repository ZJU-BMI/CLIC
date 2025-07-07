library(locuszoomr)
library(ggplot2)

library(EnsDb.Hsapiens.v75)
# patchwork method
library(patchwork)
rm(list=ls())

source('/Users/natsumikyouno/ECG/codes/genetic_analysis/gg_scatter.R')

df_leads <- read.csv('gwas_results/non_insomnia/lead_snp.csv')
snps <- df_leads$SNP
genes <- df_leads$Nearest.GENE
chrs <- df_leads$CHR
distance_ld <- 500000

# Create the directory if it doesn't exist
save_path <- 'gwas_results/non_insomnia/regional_plot'
if (!dir.exists(save_path)) {
    dir.create(save_path, recursive = TRUE)
}

i <- 5
for (i in 1:length(snps)) {
    snp <- snps[i]
    gene <- genes[i]
    chr <- chrs[i]
    
    print(paste0("Processing SNP: ", snp, " Gene: ", gene))
    region_path <- sprintf('gwas_results/non_insomnia/regions/%s.csv', snp)
    
    df_region <- read.csv(region_path)
    df_region <- subset(df_region, select = c('CHR', 'SNP', 'BP', 'P', 'BETA', 'SE'))
    colnames(df_region) <- c('chrom', 'rsid', 'pos', 'p', 'beta', 'se')
    
    if (require(EnsDb.Hsapiens.v75)) {
      loc <- locus(df_region, gene = gene,
                   flank = c(distance_ld, distance_ld), 
                   ens_db = "EnsDb.Hsapiens.v75")
      loc <- link_LD(loc, token = "19b0a7d7cbe2", pop = "EUR")
    }
    
    # save to jpeg
    p <- ggscatter(loc, labels = c(snp), legend_pos = 'topright',
                   size = 5, cex.axis = 2.4, cex.lab = 3, label.size = 7, shape_values = c(21, 24, 25)) +
      labs(title= '')
    theme_scatter <- theme(
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      axis.title.x = element_blank(),
      axis.line = element_line(colour = "black", linewidth = 1),
      axis.title.y = element_text(margin = margin(t = 0, r = -5, b = 0, l = 0)),
      # title
      plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
      # legend
      legend.background = element_rect(fill = "transparent", colour = NA),
      legend.text = element_text(size = 16),
      legend.title = element_text(size = 16),
      legend.key.spacing.y = unit(0.1, "cm"),
      # label text size
      text = element_text(size = 16),
    )
    p <- p + theme_scatter
    genetrack <- gg_genetracks(loc, maxrows = 4, filter_gene_biotype = 'protein_coding',
                               gene_col = 'grey', exon_col = 'red', exon_border = 'darkgrey',
                               cex.axis = 2.4, cex.text = 1.5, cex.lab = 3
    ) + theme(
      axis.line = element_line(colour = "black", linewidth = 1),
    )
    
    p  / genetrack +
      plot_layout(ncol = 1, heights = c(3, 1)) +
      theme(plot.tag = element_text(size = 20, face = "bold"))
    # Save the plot
    save_name <- sprintf("gwas_results/non_insomnia/regional_plot/%s_%s_%s.jpg", chr, snp, gene)
    ggsave(save_name, width = 9, height = 10, units = "in", dpi = 300)


}
