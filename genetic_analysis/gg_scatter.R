library(ggplot2)
library(ggrepel)

ggscatter <- function (loc, index_snp = loc$index_snp, pcutoff = 5e-08, scheme = c("grey", 
  "dodgerblue", "red"), size = 2, cex.axis = 1, cex.lab = 1, label.size = 5,
  xlab = NULL, ylab = NULL, ylim = NULL, ylim2 = c(0, 100), 
  yzero = (loc$yvar == "logP"), xticks = TRUE, border = FALSE, 
  showLD = TRUE, LD_scheme = c("grey", "royalblue", "cyan2", 
    "green3", "orange", "red", "purple"), recomb_col = "blue", 
  recomb_offset = 0, legend_pos = "topleft", labels = NULL, 
  eqtl_gene = NULL, beta = NULL, shape = NULL, shape_values = c(21, 
    24, 25), ...) 
{
  if (!inherits(loc, "locus")) 
    stop("Object of class 'locus' required")
  if (is.null(loc$data)) 
    stop("No data points, only gene tracks")
  .call <- match.call()
  data <- loc$data
  if (is.null(xlab) & xticks) 
    xlab <- paste("Chromosome", loc$seqname, "(Mb)")
  if (is.null(ylab)) {
    ylab <- if (loc$yvar == "logP") 
      expression("-log"[10] ~ "P")
    else loc$yvar
  }
  hasLD <- "ld" %in% colnames(data)
  if (!"bg" %in% colnames(data)) {
    if (showLD & hasLD) {
      data$bg <- cut(data$ld, -1:6/5, labels = FALSE)
      data$bg[is.na(data$bg)] <- 1L
      data$bg[data[, loc$labs] %in% index_snp] <- 7L
      data$bg <- factor(data$bg, levels = 1:7)
      data <- data[order(data$bg), ]
      scheme <- rep_len(LD_scheme, 7)
      if (is.null(index_snp)) {
        scheme <- scheme[1:6]
        data$bg <- factor(data$bg, levels = 1:6)
      }
    }
    else if (!is.null(eqtl_gene)) {
      bg <- data[, eqtl_gene]
      bg[data[, loc$p] > pcutoff] <- "ns"
      bg <- relevel(factor(bg, levels = unique(bg)), "ns")
      if (is.null(.call$scheme)) 
        scheme <- eqtl_scheme(nlevels(bg))
      data$bg <- bg
    }
    else {
      data$bg <- scheme[1]
      if (loc$yvar == "logP") 
        data$bg[data[, loc$p] < pcutoff] <- scheme[2]
      data$bg[data[, loc$labs] %in% index_snp] <- scheme[3]
      data$bg <- factor(data$bg, levels = scheme)
    }
  }
  if (!"col" %in% colnames(data)) 
    data$col <- "black"
  data$col <- as.factor(data$col)
  if (!is.null(shape)) {
    if (!is.null(beta)) 
      stop("cannot set both `shape` and `beta`")
    if (!shape %in% colnames(data)) 
      stop("incorrect column name for `shape`")
    shape_breaks <- shape_labels <- levels(data[, shape])
  }
  if (!is.null(beta)) {
    data[, beta] <- signif(data[, beta], 3)
    symbol <- as.character(sign(data[, beta]))
    ind <- data[, loc$p] > pcutoff
    symbol[ind] <- "ns"
    data$.beta <- factor(symbol, levels = c("ns", "1", "-1"), 
      labels = c("ns", "up", "down"))
    shape <- ".beta"
    shape_breaks <- c("ns", "up", "down")
    shape_labels <- c("ns", expression({
      beta > 0
    }), expression({
      beta < 0
    }))
  }
  legend.justification <- NULL
  legend_labels <- legend_title <- NULL
  legend.position <- "none"
  if (!is.null(legend_pos)) {
    if (legend_pos == "topleft") {
      legend.justification <- c(0, 1)
      legend.position <- c(0.01, 0.99)
    }
    else if (legend_pos == "topright") {
      legend.justification <- c(1, 1)
      legend.position <- c(0.99, 0.99)
    }
    else {
      legend.position <- legend_pos
    }
    if (showLD & hasLD) {
      legend_title <- expression({
        r^2
      })
      legend_labels <- rev(c("Index SNP", "0.8 - 1.0", 
        "0.6 - 0.8", "0.4 - 0.6", "0.2 - 0.4", "0.0 - 0.2", 
        "NA"))
      if (is.null(index_snp)) 
        legend_labels <- legend_labels[1:6]
    }
    else if (!is.null(eqtl_gene)) {
      legend_labels <- levels(bg)
    }
    else if (is.null(beta) & is.null(shape)) 
      legend.position <- "none"
  }
  yrange <- if (is.null(ylim)) 
    range(data[, loc$yvar], na.rm = TRUE)
  else ylim
  if (is.null(ylim) && yzero) 
    yrange[1] <- min(c(0, yrange[1]))
  ycut <- -log10(pcutoff)
  recomb <- !is.null(loc$recomb) & !is.na(recomb_col)
  if (recomb) {
    df <- loc$recomb[, c("start", "value")]
    colnames(df) <- c(loc$pos, "recomb")
    data <- dplyr::bind_rows(data, df)
    data <- data[order(data[, loc$pos]), ]
    data$recomb <- zoo::na.approx(data$recomb, data[, loc$pos], 
      na.rm = FALSE)
    ymult <- 100/diff(yrange)
    yd <- diff(yrange)
    yd2 <- diff(ylim2)
    yrange0 <- yrange
    yrange[1] <- yrange[1] - yd * recomb_offset
    outside <- df$recomb < ylim2[1] | df$recomb > (ylim2[2] + 
      yd2 * recomb_offset)
    if (any(outside)) 
      nmessage(sum(outside), " recombination value(s) outside scale range (`ylim2`)")
    fy2 <- function(yy) (yy - ylim2[1])/yd2 * yd + yrange[1]
    inv_fy2 <- function(yy) (yy - yrange[1])/yd * yd2 + 
      ylim2[1]
  }
  outside <- loc$data[, loc$yvar] < yrange[1] | loc$data[, 
    loc$yvar] > yrange[2]
  if (any(outside)) 
    nmessage(sum(outside), " value(s) outside scale range (`ylim`)")
  data[, loc$pos] <- data[, loc$pos]/1e+06
  if (!is.null(labels)) {
    i <- grep("index", labels, ignore.case = TRUE)
    if (length(i) > 0) {
      if (length(index_snp) == 1) {
        labels[i] <- index_snp
      }
      else {
        labels <- labels[-i]
        labels <- c(index_snp, labels)
      }
    }
    text_label_ind <- match(labels, data[, loc$labs])
    if (any(is.na(text_label_ind))) {
      message("label ", paste(labels[is.na(text_label_ind)], 
        collapse = ", "), " not found")
    }
  }
  ind <- data[, loc$labs] %in% index_snp
  if (!recomb) {
    if (is.null(shape)) {
      p <- ggplot(data[!ind, ], aes(x = .data[[loc$pos]], 
        y = .data[[loc$yvar]], color = .data$col, fill = .data$bg)) + 
        (if (loc$yvar == "logP" & !is.null(pcutoff) & 
          ycut >= yrange[1] & ycut <= yrange[2]) {
          geom_hline(yintercept = ycut, colour = "grey", 
            linetype = "dashed")
        }) + geom_point(shape = 21, size = size, color = 'white') + (if (any(ind)) {
        geom_point(data = data[ind, ], aes(y = .data[[loc$yvar]], # color = 'transparent', 
          color = .data$col, 
          fill = .data$bg), shape = 23, 
          size = size)
      })
    }
    else {
      p <- ggplot(data, aes(x = .data[[loc$pos]], y = .data[[loc$yvar]], 
        color = .data$col, fill = .data$bg, shape = .data[[shape]])) + 
        (if (loc$yvar == "logP" & !is.null(pcutoff) & 
          ycut >= yrange[1] & ycut <= yrange[2]) {
          geom_hline(yintercept = ycut, colour = "grey", 
            linetype = "dashed")
        }) + geom_point(size = size, color = 'transparent') + scale_shape_manual(values = shape_values, 
        name = NULL, breaks = shape_breaks, labels = shape_labels) + 
        (if (showLD & hasLD) {
          guides(fill = guide_legend(override.aes = list(shape = 21), 
            reverse = TRUE, order = 1))
        }
        else {
          guides(fill = "none")
        })
    }
    
    p <- p + scale_fill_manual(breaks = levels(data$bg), 
      values = scheme, guide = guide_legend(reverse = TRUE), 
      labels = legend_labels, name = legend_title) + scale_color_manual(breaks = levels(data$col), 
      values = levels(data$col), guide = "none") + xlim(loc$xrange[1]/1e+06, 
      loc$xrange[2]/1e+06) + ylim(yrange) + labs(x = xlab, 
      y = ylab) + theme_classic() + theme(axis.text = element_text(colour = "black", 
      size = 10 * cex.axis), axis.title = element_text(size = 10 * 
      cex.lab), legend.justification = legend.justification, 
      legend.position = legend.position, legend.title.align = 0.5, 
      legend.text.align = 0, legend.key.size = unit(0.9, 
        "lines"), legend.spacing.y = unit(0, "lines")) + 
      if (!xticks) 
        theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
  }
  else {
    if (is.null(shape)) {
      p <- ggplot(data[!ind, ], aes(x = .data[[loc$pos]])) + 
        (if (loc$yvar == "logP" & !is.null(pcutoff) & 
          ycut >= yrange[1] & ycut <= yrange[2]) {
          geom_hline(yintercept = ycut, colour = "grey", 
            linetype = "dashed")
        }) + geom_point(aes(y = .data[[loc$yvar]], color = 'transparent', # color = .data$col, 
        fill = .data$bg), shape = 21, size = size, na.rm = TRUE) + 
        (if (any(ind)) {
          geom_point(data = data[ind, ], aes(y = .data[[loc$yvar]], color = 'transparent', 
            # color = .data$col, 
            fill = .data$bg), shape = 23, 
            size = size, na.rm = TRUE)
        })
    }
    else {
      p <- ggplot(data, aes(x = .data[[loc$pos]])) + (if (loc$yvar == 
        "logP" & !is.null(pcutoff) & ycut >= yrange[1] & 
        ycut <= yrange[2]) {
        geom_hline(yintercept = ycut, colour = "grey", 
          linetype = "dashed")
      }) + geom_point(aes(y = .data[[loc$yvar]], color = 'transparent', # color = .data$col, 
        fill = .data$bg, shape = .data[[shape]]), size = size, 
        na.rm = TRUE) + scale_shape_manual(values = shape_values, 
        name = NULL, breaks = shape_breaks, labels = shape_labels) + 
        (if (showLD & hasLD) {
          guides(fill = guide_legend(override.aes = list(shape = 21), 
            reverse = TRUE, order = 1))
        }
        else {
          guides(fill = "none")
        })
    }
    p <- p + scale_fill_manual(breaks = levels(data$bg), 
      values = scheme, guide = guide_legend(reverse = TRUE), 
      labels = legend_labels, name = legend_title) + scale_color_manual(breaks = levels(data$col), 
      values = levels(data$col), guide = "none") + geom_line(aes(y = fy2(.data$recomb)), 
      color = recomb_col, na.rm = TRUE) + scale_y_continuous(name = ylab, 
      limits = yrange, breaks = pretty(yrange0), sec.axis = sec_axis(inv_fy2, 
        name = "Recombination rate (%)", breaks = pretty(ylim2))) + 
      xlim(loc$xrange[1]/1e+06, loc$xrange[2]/1e+06) + 
      xlab(xlab) + theme_classic() + theme(axis.text = element_text(colour = "black", 
      size = 10 * cex.axis), axis.title = element_text(size = 10 * 
      cex.lab), axis.title.y.left = element_text(hjust = min(c(0.5 + 
      recomb_offset/3, 0.9))), axis.title.y.right = element_text(hjust = min(c(0.5 + 
      recomb_offset/2, 1))), legend.justification = legend.justification, 
      legend.position = legend.position, legend.title.align = 0.5, 
      legend.text.align = 0, legend.key.size = unit(0.9, 
        "lines"), legend.spacing.y = unit(0, "lines")) + 
      if (!xticks) 
        theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
  }
  if (!is.null(labels)) {
    p <- p + geom_text_repel(data = data[text_label_ind, 
      ], mapping = aes(x = .data[[loc$pos]], y = .data[[loc$yvar]], 
      label = .data[[loc$labs]]), size = label.size, point.size = size, ...)
  }
  if (border | recomb) {
    p <- p + theme(panel.border = element_rect(colour = "black", 
      fill = NA))
  }
  p
}
