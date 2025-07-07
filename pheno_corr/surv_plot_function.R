library(survival)
library(survminer)
library(ggplot2)
library(patchwork)
library(tools)

show_p <- function(pval) {
  if (pval < 0.001) {
    pmarker <- '***'
  } else if (pval < 0.01) {
    pmarker <- '**'
  } else if (pval < 0.05) {
    pmarker <- '*'
  } else {
    pmarker <- ''
  }
  print(paste('pval: ', pval, ' ', pmarker))
  if (pval < 2.23e-308) {
    res_str <- paste0('< 2.23e-308', pmarker)
  } else {
    res_str <- paste0(as.character(signif(pval, digits = 2)), pmarker)
  }
  return(res_str)
}


survplot <- function(fit, df, linetype, color_palette, legend_labs,
                     xlim = c(0, 10),
                     break_x = 2,
                     min_y = 0.90,
                     break_y = 0.02,
                     annot_pos = 0.95,
                     font_size = 5.4,
                     legend_pos = c(0.75, 0.88),
                     fig_save_path = 'results/survival_curve/survival_plot.jpg',
                     pval_str = '',
                     pval_size = 5.4,
                     title = '') {
  # plot survival curve
  psurv <- ggsurvplot(fit, # 创建的拟合对象
                      data = df,  # 指定变量数据来源
                      conf.int = TRUE, # 显示置信区间
                      xlim = xlim, # 设置x轴范围
                      ylim = c(min_y, 1), # 设置y轴范围
                      linetype = linetype, # 设置线型
                      # surv.median.line = "hv",  # 添加中位生存时间线
                      title = title, # 设置标题
                      xlab = "Time (years)",
                      ylab = "Survival Probility",
                      # legend = c(0.1, 0.12), # 指定图例位置
                      legend.title = "", # 设置图例标题，这里设置不显示标题，用空格替代
                      legend.labs = legend_labs, # 指定图例分组标签
                      break.x.by = break_x,  # 设置x轴刻度间距
                      break.y.by = break_y, # 设置y轴刻度间距
                      palette = color_palette, # 设置颜色
                      risk.table = T, # 显示risk.table
                      risk.table.title = "Number at risk", # risk.table标题
                      # risk.table.fontsize = 4.5, # risk.table字体大小
                      risk.table.col = "strata", # risk.table文字颜色，strata与曲线图一致
                      risk.table.y.text = F, # risk.table y轴显示文字，FALSE则显示色块
                      risk.table.height = 0.25, # risk.table占图片比例
                      surv.plot.height = 0.75, # 生存图占图片比例
                      pval = F, # 显示p值
                      tables.y.text = F, # tables y轴显示文字，FALSE则显示色块
                      cumevents = T, # 显示累积事件数
                      cumevents.title = 'Number of events', # 累积事件数标题
                      fontsize = 5.4, # 字体大小
                      cumevents.col = "strata",
                      cumcensor = F, # 显示累积censor数
                      censor = F, # 显示censor
  )
  
  # theme for survival plot
  theme_surv <- theme(
    plot.title = element_text(hjust = 0.5, size=20),
    axis.text.x = element_text(hjust = 0.5, size=22), 
    axis.text.y = element_text(size = 22),
    axis.title.x = element_text(size = 22), 
    axis.title.y = element_text(size = 22), 
    
    legend.text = element_text(size = 20),
    # legend.title = element_blank(),
    legend.position = legend_pos,
    legend.background = element_blank(),
    # spacing between each legend key
    legend.key.size = unit(1.2, "lines"),
    legend.key.spacing.y = (unit(0.2, "cm")), # 设置图例间距
    # remove top and right border of the plot
    # plot border linewidth 1.5
    # panel.border = element_rect(linewidth = 1.5, fill = NA),
    panel.border = element_blank(),
    panel.grid = element_blank(),
    axis.line = element_line(color = "black", linewidth = 1.5), # 保留X轴线
  )
  
  theme_tab <-  theme(
    plot.title = element_text(hjust = 0., size = 24),
    axis.text.x = element_text(hjust = 0.5, size = 22),
    axis.title.x = element_text(size = 24), 
    panel.border = element_blank(),
    panel.grid = element_blank(),
    axis.line = element_line(color = "black", linewidth = 1.5), # 保留X轴线
    axis.ticks.x = element_line(color = "black", linewidth = 1), # 保留X轴刻度
    axis.ticks.y = element_blank(),
    legend.position = "none"
  )
  
  psurv$plot <- psurv$plot + theme_bw() + theme_surv + 
    annotate("text", x = 0., y = annot_pos,
             label= pval_str, size = 5.4, hjust = 0) 
  
  psurv$table <- psurv$table + theme_tab
  psurv$cumevents <- psurv$cumevents + theme_tab
  
  # psurv
  
  # remove x-axis on the survival plot
  theme2 <- theme(axis.ticks.x = element_blank(), ## 删去所有刻度线
                  axis.text.x = element_blank(),
                  axis.title.x = element_blank(),
  )
  plot.up <- psurv$plot 
  
  plot.mid <- psurv$table + theme2
  plot.down <- psurv$cumevents + theme2
  plot.up / plot_spacer() / plot.mid / plot_spacer() / plot.down + 
    plot_layout(heights = c(6, -0.3, 1.2, -0.2, 1.2)) ##调整中间空白
  
  ggsave(fig_save_path, width = 8, height = 10, dpi = 300)
}