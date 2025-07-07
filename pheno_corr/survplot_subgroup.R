library(survival)
library(survminer)
library(ggplot2)
library(patchwork)
library(tools)

rm(list = ls())
source('surv_plot_function.R')

df <- read.csv('data/ukb_ecg_data_balanced_cluster.csv')

sex <- 1
if (sex != -1) {
  df <- subset(df, Sex == sex)
}

# copy label
df$label <- df$event

# fit survival model
# 00, 01, 10, 11 -> insomnia T+, non-insomnia T+, insomnia T-, non-insomnia T-
fit <- survfit(Surv(time, label) ~ cluster_assign + insomnia_score, data=df)
# p-value for T+ vs. T-
p <- surv_pvalue(fit)$pval

# pval insomnia T+ vs. non-insomnia T+
pval_str1 <- show_p(surv_pvalue(fit, data = df[df$cluster_assign == 0, ])$pval)
# pval insomnia T- vs. non-insomnia T-
pval_str2 <- show_p(surv_pvalue(fit, data = df[df$cluster_assign == 1, ])$pval)
# pval insomnia T+ vs. insomnia T-
pval_str3 <- show_p(surv_pvalue(fit, data= df[df$insomnia_score == 1, ])$pval)
# pval non-insomnia T+ vs. non-insomnia T-
pval_str4 <- show_p(surv_pvalue(fit, data= df[df$insomnia_score == 0, ])$pval)

pval_str <- paste('LogRank Test:', 
                  paste('Insomnia T+ vs. Non-insomnia T+: P =', pval_str1, sep = ' '),
                  paste('Insomnia T- vs. Non-insomnia T-: P =', pval_str2, sep = ' '),
                  # paste('Insomnia T+ vs. T-: P = ', pval_str3, sep = ' '),
                  # paste('Non-insomnia T+ vs. T-: P =', pval_str4, sep = ' '),
                  sep = '\n')

title <- ''
color_palette <- c('salmon', '#D6101E', '#4ebbd8', '#0d6ea3')
linetype = c("dashed", "solid", "dashed", "solid")
legend_labs = c("Insomnia T+","Non-insomnia T+", 'Insomnia T-', 'Non-insomnia T-')
legend_pos = c(0.75, 0.88)
xlim = c(0, 10)
break_x = 2 
min_y <- 0.86
break_y = 0.02
annot_pos <- min_y + (1 - min_y) * 0.1

if (sex == -1) {
  fig_save_path <- sprintf('results/survival_curve/survival_plot_case_control_subtype.pdf')
} else {
  fig_save_path <- sprintf('results/survival_curve/survival_plot_case_control_subtype_%s.jpg', sex)
}

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



