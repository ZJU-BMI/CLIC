library(survival)
library(survminer)
library(ggplot2)
library(patchwork)
library(tools)

rm(list = ls())
# Read data
source('surv_plot_function.R')

df <- read.csv('data/ukb_ecg_data_balanced_cluster.csv')

# copy label
df$label <- df$event

# fit survival model
fit <- survfit(Surv(time, label) ~ insomnia_score, data=df)

# p-value for T+ vs. T-
p <- surv_pvalue(fit)$pval

pval <- show_p(p)
pval_str <- paste('LogRank P-value =', pval, sep = ' ')
print(pval_str)

title <- ''
color_palette <-  c('#f4b183', '#8faadc')
title <- ''
linetype <- c("dashed", "solid") 
legend_labs <- c("Insomnia", 'Non-insomnia')
legend_pos <- c(0.75, 0.95)
xlim <- c(0, 10) #
break_x <- 2 
min_y <- 0.92
break_y = 0.02 # 设置y轴刻度间距
annot_pos <- min_y + (1 - min_y) * 0.1
fig_save_path <- sprintf('results/survival_curve/survival_plot_insomnia.pdf')


# plot survival curve
survplot(fit, df, linetype, color_palette, legend_labs,
         xlim = xlim,
         break_x = break_x,
         min_y = min_y,
         break_y = break_y,
         annot_pos = annot_pos,
         font_size = 5.4,
         legend_pos = legend_pos,
         fig_save_path = fig_save_path,
         pval_str = pval_str,
         pval_size = 5.6,
         title = title)


