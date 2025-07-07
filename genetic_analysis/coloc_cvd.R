library(coloc)
library(dplyr)
library(readxl)
library(fs)

res_path <- 'results/coloc_results'
if (!dir_exists(res_path)) {
  dir_create(res_path)
}

# exposure

# outcome gwas id (from opengwas)
y_gwas_id <- read.csv('gwas_data/cvd_gwas.csv')

pheno_x <- 'ECG+'
pheno_y <- y_gwas_id$pheno_abbv
x_gwas_files <- 'res.txt.gz'
pheno_y_name <- y_gwas_id$pheno

sample_size_x <- 40360
sample_size_y <- y_gwas_id$N
case_x <- 24889
case_y <- y_gwas_id$N_cases

x_gwas_path <- '/Users/natsumikyouno/ECG/data/GWAS/ecg_noninsomnia'
y_gwas_path <- '/Users/natsumikyouno/ECG/data/GWAS/CVD'

# read gwas files
i <- 1

coloc_res_path <- 'results/coloc_results/ecg_noninsomnia_cvd.csv'
if (file_exists(coloc_res_path)) {
  coloc_results <- read.csv(coloc_res_path)
} else {
  coloc_results <- read.csv('results/coloc_results/coloc_example.csv')
}


x_gwas <- read.csv(file.path(x_gwas_path, x_gwas_files), sep = ' ')
# keep only the columns we need to save memory
x_gwas <- x_gwas[, c('SNP', 'P', 'BETA', 'SE', 'BP', 'CHR')]


x_leads <- read.csv(file.path(x_gwas_path, 'lead_snp.csv'))
i <- 5


s_x <- case_x / sample_size_x

for (i in 1:nrow(x_leads)) {
    
    lead_snp_i <- x_leads$SNPID[i]
    bp_i <- x_leads$POS[i]
    chr_i <- x_leads$CHR[i]
    min_bp <- bp_i - 250000
    max_bp <- bp_i + 250000
    print(paste('lead snp', lead_snp_i, 'at region [', min_bp, bp_i, max_bp, ']'))
    x_regions <- subset.data.frame(x_gwas, CHR == chr_i & BP > min_bp & BP < max_bp)
    print(nrow(x_regions))
    if (nrow(x_regions) < 100){
      next
    }
    for (j in 1:length(pheno_y)) {
      
      # check if results already exist
      coloc_xy <- coloc_results %>% filter(x == pheno_x & y == pheno_y_name[j] & lead_snp == lead_snp_i)
      if (nrow(coloc_xy) > 0) {
        print(paste0('Coloc results for ', pheno_x, ' and ', pheno_y_name[j], ' already exist. Skipping...'))
        next
      }
      
      print(paste0('Running coloc for ', pheno_x, ' and ', pheno_y_name[j]))
      y <- pheno_y[j]
      y_gwas <- read.csv(file.path(y_gwas_path, paste0(pheno_y[j], '.txt.gz')), sep = ' ')
      y_gwas <- y_gwas[, c('SNP', 'P', 'BETA', 'SE')]
      
      s_y <- case_y[j] / sample_size_y[j]
      
      input <- merge(x_regions, y_gwas, by = 'SNP', all = FALSE, suffixes=c("_x", "_y"))
      input <- na.omit(input)
      # drop duplicate SNPs
      input <- input[!duplicated(input$SNP), ]
      result <- coloc.abf(dataset1=list(pvalues = input$P_x,
                                        type = "cc", 
                                        N = 40360,
                                        beta = input$BETA_x,
                                        varbeta = input$SE_x^2,
                                        S = s_x, 
                                        snp = input$SNP),
                          
                          dataset2=list(pvalues=input$P_y,
                                        type="cc", 
                                        beta = input$BETA_y,
                                        varbeta = input$SE_y^2,
                                        N = sample_size_y[j],
                                        S = s_y, 
                                        snp = input$SNP))
      
      coloc_result <- result$results %>% filter(SNP.PP.H4 >= 0.75)
      # write.csv(coloc_result, 'coloc_results/coloc_example.csv', row.names = F)
      if (nrow(coloc_result) > 0) {
        print(paste0(nrow(coloc_result), ' SNPs with PP.H4 >= 0.75 for ', pheno_x, ' and ', y, ' lead SNP', lead_snp_i))
        coloc_result$x <- pheno_x
        coloc_result$y <- pheno_y_name[j]
        coloc_result$lead_snp <- lead_snp_i
        coloc_results <- rbind.data.frame(coloc_results, coloc_result)
        
      } else {
        # set the first row of coloc_result to NA
        coloc_result[1, ] <- NA
        coloc_result$x <- pheno_x
        coloc_result$y <- pheno_y_name[j]
        coloc_result$lead_snp <- lead_snp_i
        coloc_results <- rbind.data.frame(coloc_results, coloc_result)
        print(paste0('No SNPs with PP.H4 >= 0.75 for ', pheno_x, ' and ', y, ' lead SNP', lead_snp_i))
      }
      # save results
      write.csv(coloc_results, coloc_res_path, row.names = F)
    }
}


