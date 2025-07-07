library("locuszoomr")
scatter_plot <- function (loc, index_snp = loc$index_snp, pcutoff = 5e-08, 
                          scheme = c("grey", "dodgerblue", "red"), 
                          cex = 1, cex.axis = 0.9, cex.lab = 1, cex.legend = 1,
          xlab = NULL, ylab = NULL, yzero = (loc$yvar == "logP"), 
          xticks = TRUE, border = FALSE, showLD = TRUE, 
          LD_scheme = c("grey", "royalblue", "cyan2", "green3", "orange", "red", "purple"), 
          recomb_col = "blue", legend_pos = "topleft", labels = NULL, 
          label_x = 4, label_y = 4, eqtl_gene = NULL, beta = NULL, 
          add = FALSE, align = TRUE, ...) 
{
  if (!inherits(loc, "locus")) 
    stop("Object of class 'locus' required")
  if (is.null(loc$data)) 
    stop("No data points, only gene tracks")
  .call <- match.call()
  data <- loc$data
  if (is.null(xlab)) 
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
      data <- data[order(data$bg), ]
      LD_scheme <- rep_len(LD_scheme, 7)
      data$bg <- LD_scheme[data$bg]
    }
    else if (!is.null(eqtl_gene)) {
      bg <- data[, eqtl_gene]
      bg[data[, loc$p] > pcutoff] <- "ns"
      bg <- relevel(factor(bg, levels = unique(bg)), "ns")
      if (is.null(.call$scheme)) 
        scheme <- eqtl_scheme(nlevels(bg))
      data$bg <- scheme[bg]
    }
    else {
      data$bg <- 1L
      if (loc$yvar == "logP") 
        data$bg[data[, loc$p] < pcutoff] <- 2L
      data$bg[data[, loc$labs] %in% index_snp] <- 3L
      data <- data[order(data$bg), ]
      data$bg <- scheme[data$bg]
    }
  }
  recomb <- !is.null(loc$recomb) & !is.na(recomb_col)
  if (align) {
    op <- par(mar = c(ifelse(xticks, 3, 0.1), 3.5, 2, ifelse(recomb, 
                                                             3.5, 1.5)))
    on.exit(par(op))
  }
  ylim <- range(data[, loc$yvar], na.rm = TRUE)
  if (yzero) 
    ylim[1] <- min(c(0, ylim[1]))
  if (!is.null(labels) & (border | recomb)) {
    ylim[2] <- ylim[2] + diff(ylim) * 0.08
  }
  panel.first <- quote({
    if (loc$yvar == "logP" & !is.null(pcutoff)) {
      abline(h = -log10(pcutoff), col = "darkgrey", lty = 2)
    }
    if (recomb) {
      ry <- loc$recomb$value * diff(ylim)/100 + ylim[1]
      lines(loc$recomb$start, ry, col = recomb_col)
      at <- 0:5 * (diff(ylim)/5) + ylim[1]
      axis(4, at = at, labels = 0:5 * 20, las = 1, tcl = -0.3, 
           mgp = c(1.7, 0.5, 0), cex.axis = cex.axis)
      mtext("Recombination rate (%)", 4, cex = cex.lab, 
            line = 1.7)
    }
  })
  pch <- rep(21L, nrow(data))
  pch[data[, loc$labs] %in% index_snp] <- 23L
  if (!is.null(beta)) {
    sig <- data[, loc$p] < pcutoff
    pch[sig] <- 24 + (1 - sign(data[sig, beta]))/2
  }
  if ("pch" %in% colnames(data)) 
    pch <- data$pch
  col <- "black"
  if ("col" %in% colnames(data)) 
    col <- data$col
  if ("cex" %in% colnames(data)) 
    cex <- data$cex
  new.args <- list(...)
  if (add) {
    plot.args <- list(x = data[, loc$pos], y = data[, loc$yvar], 
                      pch = pch, bg = data$bg, cex = cex)
    if (length(new.args)) 
      plot.args[names(new.args)] <- new.args
    return(do.call("points", plot.args))
  }
  bty <- if (border | recomb) 
    "o"
  else "l"
  plot.args <- list(x = data[, loc$pos], y = data[, loc$yvar], 
                    pch = pch, bg = data$bg, col = col, las = 1, font.main = 1, 
                    cex = cex, cex.axis = cex.axis, cex.lab = cex.lab, xlim = loc$xrange, 
                    ylim = ylim, xlab = if (xticks) xlab else "", ylab = ylab, 
                    bty = bty, xaxt = "n", tcl = -0.3, mgp = c(1.7, 0.5,0), 
                    panel.first = panel.first, lwd = 5)
  if (length(new.args)) 
    plot.args[names(new.args)] <- new.args
  do.call("plot", plot.args)
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
    ind <- match(labels, data[, loc$labs])
    if (any(is.na(ind))) {
      message("label ", paste(labels[is.na(ind)], collapse = ", "), 
              " not found")
    }
    lx <- data[ind, loc$pos]
    ly <- data[ind, loc$yvar]
    labs <- data[ind, loc$labs]
    add_labels(lx, ly, labs, label_x, label_y, cex = cex.axis * 
                 0.95)
  }
  if (xticks) {
    axis(1, at = axTicks(1), labels = axTicks(1)/1e+06, 
         cex.axis = cex.axis, mgp = c(1.7, 0.4, 0), tcl = -0.3)
  }
  else if (!border) {
    axis(1, at = axTicks(1), labels = FALSE, tcl = -0.3)
  }
  if (!is.null(legend_pos)) {
    if (!is.null(eqtl_gene) | !is.null(beta)) {
      leg <- pt.bg <- pch <- NULL
      if (!is.null(eqtl_gene)) {
        leg <- levels(bg)[-1]
        pt.bg <- scheme[-1]
        pch <- c(rep(21, length(scheme) - 1))
      }
      if (!is.null(beta)) {
        leg <- c(leg, expression({
          beta > 0
        }), expression({
          beta < 0
        }))
        pch <- c(pch, 2, 6)
        pt.bg <- c(pt.bg, NA)
      }
      legend(legend_pos, legend = leg, y.intersp = 0.96, 
             pch = pch, pt.bg = pt.bg, col = "black", bty = "n", 
             cex = cex.legend)
    }
    else if (showLD & hasLD) {
      legend(legend_pos, legend = c("0.8 - 1.0", "0.6 - 0.8", 
                                    "0.4 - 0.6", "0.2 - 0.4", "0.0 - 0.2"), title = expression({
                                      r^2
                                    }), y.intersp = 0.96, pch = 21, col = "black", pt.bg = rev(LD_scheme[-c(1, 
                                                                                                            7)]), bty = "n", cex = cex.legend)
    }
  }
}



add_labels <- function (lx, ly, labs, label_x, label_y, cex = 1) 
{
  label_x <- rep_len(label_x, length(lx))
  label_y <- rep_len(label_y, length(ly))
  dx <- diff(par("usr")[1:2]) * label_x/100
  dy <- diff(par("usr")[3:4]) * label_y/100
  dlines(lx, ly, dx, dy, xpd = NA)
  adj1 <- -sign(dx) * 0.56 + 0.5
  adj2 <- -sign(dy) + 0.5
  adj2[abs(label_x) > abs(label_y)] <- 0.5
  adj1[abs(label_x) < abs(label_y)] <- 0.5
  if (length(unique(adj1)) == 1 & length(unique(adj2)) == 
      1) {
    adj <- c(adj1[1], adj2[1])
    text(lx + dx, ly + dy, labs, adj = adj, cex = cex, xpd = NA)
  }
  else {
    adj <- cbind(adj1, adj2)
    for (i in seq_along(labs)) {
      text(lx[i] + dx[i], ly[i] + dy[i], labs[i], adj = adj[i, 
      ], cex = cex, xpd = NA)
    }
  }
}

dlines <- function(x, y, dx, dy, ...) {
  mx <- cbind(x, x + dx, NA)
  my <- cbind(y, y + dy, NA)
  xs <- as.vector(t(mx))
  ys <- as.vector(t(my))
  lines(xs, ys, ...)
}
