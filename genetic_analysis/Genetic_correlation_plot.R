rm(list = ls())
library(dplyr)
library(data.table)
library(stringr)
library(ggplot2)
library(readxl)

pheno <- 'Diseases'

df <- read.csv('results/rg_cvds.csv')
df_dis <- read.csv('gwas_data/cvd_gwas.csv')
df <- merge(df, df_dis, by.x = 'p2', by.y = 'pheno', all.x = TRUE)

# remove NA
df <- df[df$rg != 'NA', ]
df$rg <- as.numeric(df$rg)
df$se <- as.numeric(df$se)

df$sig <- ifelse(df$p < 0.05, 'p < 0.05', 'NA')
color_pal <- c('#75a4c9', '#9d69b1')

df$rg_label <- formatC(df$rg, format = 'f', digits = 2)


p <- ggplot(df, aes(x = pheno_abv, y = rg, color = p1)) + 
  geom_errorbar(aes(ymin = rg - se, ymax = rg + se), 
                position = position_dodge(width = 0.5), width = 0, size = 1) + 
  geom_point(size = 3, position = position_dodge(width = 0.5)) +
  
  geom_text(aes(label = rg_label), nudge_y = df$se + 0.03) + 
  
  scale_color_manual(values = alpha(color_pal, 0.8)) +
  scale_shape_manual(values = c(16, 17)) +
  xlab('') +
  ylab("Genetic correlation") +
  expand_limits(y = 0) + 
  scale_y_continuous(breaks = seq(-1, 1, 0.2)) +
  
  theme(plot.margin = unit(rep(1,4),'lines')) +
  # theme(panel.background = element_blank()) +  
  theme_classic() + 
  theme(axis.line = element_line(size = 0.8, colour = "black")) +
  geom_hline(yintercept = 0,linetype = 5, col = 'black') +
  
  theme(axis.title.y = element_text(size = 14, color = "black"),
        axis.title.x = element_text(size = 14, color = "black"),
        axis.text.x = element_text(size = 12, color = "black", angle = 45, vjust = 0.5, hjust = 0.5),
        axis.text.y = element_text(size = 12, color = "black")) + 
  theme(legend.title = element_blank(),
        # legend.direction = 'horizontal',
        legend.position = 'top',
        legend.text = element_text(size = 14),
        )

p

if (pheno == 'Disease') {
  res_path <- 'results/figs/genetic_corr_disease.jpg'
} else {
  res_path <- 'results/genetic_corr_traits.jpg'
}


ggsave(res_path, dpi = 300, width = 16, height = 6)


