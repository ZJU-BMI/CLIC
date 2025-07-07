library(locuszoomr)
library(ggplot2)
rm(list=ls())
library(EnsDb.Hsapiens.v75)
# patchwork method
library(patchwork)


df_region <- read.csv('/Users/natsumikyouno/ECG/data/GWAS/ecg_noninsomnia/CHR/chr21.txt.gz', sep = ' ')
df_region <- subset(df_region, select = c('CHR', 'SNP', 'BP', 'P', 'BETA', 'SE'))
colnames(df_region) <- c('chrom', 'rsid', 'pos', 'p', 'beta', 'se')

distance_ld <- 250000


if (require(EnsDb.Hsapiens.v75)) {
  loc <- locus(df_region, gene = 'KCNE1',
               flank = c(distance_ld, distance_ld), 
               ens_db = "EnsDb.Hsapiens.v75")
  loc <- link_LD(loc, token = "19b0a7d7cbe2", pop = "EUR")
}
loc_snps <- loc$data

gwas_dis <- read.csv('/Users/natsumikyouno/ECG/data/GWAS/CVD/CHR/Angina_chr21.txt.gz', sep = ' ')
gwas_dis <- subset(gwas_dis, select = c('CHR', 'SNP', 'BP', 'P', 'BETA', 'SE'))
colnames(gwas_dis) <- c('chrom', 'rsid', 'pos', 'p', 'beta', 'se')



if (require(EnsDb.Hsapiens.v75)) {
  loc2 <- locus(gwas_dis, gene = 'KCNE1',
               flank = c(distance_ld, distance_ld), 
               ens_db = "EnsDb.Hsapiens.v75")
  loc2 <- link_LD(loc2, token = "19b0a7d7cbe2", pop = "EUR")
}

# loc2_snps <- loc2$data

# save to jpeg
# jpeg("results/coloc_results/coloc_rs28451064_Angina.jpg", width = 7.5, height = 9, units = "in", res = 300)
# oldpar <- set_layers(2)
# 
# # region plot 1
# scatter_plot(loc, xticks = FALSE, labels = c('rs28451064', 'rs1805128'), 
#              label_y = c(4), col = "grey", 
#              legend_pos = "topright", lwd = 2,
#              # font size
#              cex = 1.6, cex.axis = 2.5, cex.lab = 3, cex.legend = 1.5)
# 
# # region plot 2
# scatter_plot(loc2, col = "orange", xticks = FALSE, labels = c('rs28451064'), 
#              legend_pos = "topright", lwd = 2,
#              cex = 1.6, cex.axis = 2.5, cex.lab = 3, cex.legend = 1.5)
# 
# # gene track
# genetracks(loc, maxrows = 3, filter_gene_biotype = 'protein_coding',
#            gene_col = 'grey', exon_col = 'red', exon_border = 'darkgrey',
#            cex.axis = 2.5, cex.text = 2, cex.lab = 3
#            )
# 
# par(oldpar)  # revert par() settings
# dev.off()

source('/Users/natsumikyouno/ECG/codes/genetic_analysis/gg_scatter.R')
p <- ggscatter(loc, labels = c( 'rs1805128'), legend_pos = 'topright',
                size = 5, cex.axis = 2.4, cex.lab = 3, label.size = 7, shape_values = c(21, 24, 25)) +
  labs(title= 'T+ Phenotype')
p1 <- ggscatter(loc2, labels = c('rs28451064'), legend_pos = 'topright', 
                 size = 5, cex.axis = 2.4, cex.lab = 3, label.size = 7, shape_values = c(21, 24, 25)) +
  labs(title = 'Angina')
theme_scatter <- theme(
                   axis.text.x = element_blank(),
                   axis.ticks.x = element_blank(),
                   axis.title.x = element_blank(),
                   axis.line = element_line(colour = "black", linewidth = 2),
                   axis.title.y = element_text(margin = margin(t = 0, r = -15, b = 0, l = 0)),
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
p1 <- p1 + theme_scatter
p <- p + theme_scatter
genetrack <- gg_genetracks(loc, maxrows = 3, filter_gene_biotype = 'protein_coding',
           gene_col = 'grey', exon_col = 'red', exon_border = 'darkgrey',
           cex.axis = 2.4, cex.text = 1.5, cex.lab = 3
           ) + theme(
             axis.line = element_line(colour = "black", linewidth = 2),
           )

p / p1 / genetrack + 
  plot_layout(ncol = 1, heights = c(2.5, 2.5, 1)) + 
  theme(plot.tag = element_text(size = 20, face = "bold"))
ggsave("results/coloc_results/coloc_rs28451064_Angina.jpg", width = 8, height = 10, units = "in", dpi = 300)


df_cad <- read.csv('/Users/natsumikyouno/ECG/data/GWAS/CVD/CHR/CAD_chr21.txt.gz', sep = ' ')
df_cad <- subset(df_cad, select = c('CHR', 'SNP', 'BP', 'P', 'BETA', 'SE'))
colnames(df_cad) <- c('chrom', 'rsid', 'pos', 'p', 'beta', 'se')
if (require(EnsDb.Hsapiens.v75)) {
  loc3 <- locus(df_cad, gene = 'KCNE1',
                flank = c(distance_ld, distance_ld), 
                ens_db = "EnsDb.Hsapiens.v75")
  loc3 <- link_LD(loc3, token = "19b0a7d7cbe2", pop = "EUR")
}
loc3_snps <- loc3$data

p2 <- ggscatter(loc3, labels = c('rs28451064'), legend_pos = 'topright', 
                 size = 5, cex.axis = 2.4, cex.lab = 3, label.size = 7, shape_values = c(21, 24, 25)) +
  labs(title = 'Coronary Artery Disease')
p2 <- p2 + theme_scatter
p / p2 / genetrack + 
  plot_layout(ncol = 1, heights = c(2.5, 2.5, 1)) + 
  theme(plot.tag = element_text(size = 20, face = "bold"))
ggsave("results/coloc_results/coloc_rs28451064_CAD.pdf", width = 8, height = 10, units = "in", dpi = 300)


# a plot with no text
p3 <- ggscatter(loc, labels = c(), legend_pos = 'topright',
                size = 5, cex.axis = 2.4, cex.lab = 3, label.size = 7, shape_values = c(21, 24, 25)) +
  labs(title= '', x = '', y = '') 
p4 <- ggscatter(loc2, labels = c(), legend_pos = 'topright', 
                 size = 5, cex.axis = 2.4, cex.lab = 3, label.size = 7, shape_values = c(21, 24, 25)) +
  labs(title = '', x = '', y = '')
theme_notext <-
  theme(
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.x = element_blank(),
    axis.title.x = element_blank(),
    axis.line = element_line(colour = "black", linewidth = 2),
    axis.title.y = element_text(margin = margin(t = 0, r = -15, b = 0, l = 0)),
    # title
    plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
    # legend
    legend.position = "none", 
    # label text size
  )
p4 <- p4 + theme_notext
p3 <- p3 + theme_notext
genetrack <- gg_genetracks(loc, maxrows = 3, filter_gene_biotype = 'protein_coding',
           gene_col = 'grey', exon_col = 'red', exon_border = 'darkgrey',
           cex.axis = 2.4, cex.text = 1.5, cex.lab = 3
           ) + theme(
             axis.line = element_line(colour = "black", linewidth = 2),
             legend.position = "none",
             axis.x.text = element_blank(),
             axis.y.text = element_blank(),
             axis.title.x = element_blank(),
             axis.title.y = element_blank(),
             axis.ticks.x = element_blank(),
             axis.ticks.y = element_blank(),
           )
p3 / p4 + genetrack +
  plot_layout(ncol = 1, heights = c(2.5, 2.5, 1.5)) + 
  theme(plot.tag = element_text(size = 20, face = "bold"))

ggsave("results/coloc_results/coloc_no_text.jpg", width = 7, height = 8.5, units = "in", dpi = 300)
