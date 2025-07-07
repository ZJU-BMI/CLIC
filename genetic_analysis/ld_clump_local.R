ld_clump_local <- function (dat, clump_kb, clump_r2, clump_p, bfile, plink_bin) 
{
  shell <- ifelse(Sys.info()["sysname"] == "Windows", "cmd", 
                  "sh")
  # fn <- tempfile()
  rand_name <- paste0("ld_clump_", format(Sys.time(), "%Y%m%d_%H%M%S"), "_", 
                        sample(1:10000, 1))
  fn <- file.path("data/tmp", rand_name)
  if (!dir.exists("data/tmp")) {
    dir.create("data/tmp")
  }
  write.table(data.frame(SNP = dat[["rsid"]], P = dat[["pval"]]), 
              file = fn, row.names = F, col.names = T, quote = F)
  
  fun2 <- paste0(shQuote(plink_bin, type = shell),
                 " --bfile ", shQuote(bfile, type = shell), 
                 " --clump ", shQuote(fn, type = shell), 
                 " --clump-p1 ", clump_p,
                 " --clump-r2 ", clump_r2, 
                 " --clump-kb ", clump_kb,
                 " --out ", shQuote(fn, type = shell))
  system(fun2)
  if (file.exists(paste(fn, ".clumped", sep = ""))) {
    res <- read.table(paste(fn, ".clumped", sep = ""), header = T)
  } else {
    res <- dat
  }
  
  unlink(paste(fn, "*", sep = ""))
  y <- subset(dat, !dat[["rsid"]] %in% res[["SNP"]])
  if (nrow(y) > 0) {
    message("Removing ", length(y[["rsid"]]), " of ", nrow(dat), 
            " variants due to LD with other variants or absence from LD reference panel")
  }
  return(subset(dat, dat[["rsid"]] %in% res[["SNP"]]))
}