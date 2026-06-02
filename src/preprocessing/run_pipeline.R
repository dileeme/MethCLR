library(minfi)
library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
library(IlluminaHumanMethylation450kmanifest)

idat_dir <- "/root/MethCLR_local/GSE85210/"
output_dir <- "/root/gdrive/MethCLR/data/GSE85210/beta_matrices/"
dir.create(output_dir, showWarnings=FALSE, recursive=TRUE)

samples <- c(); labels <- c()
con <- gzcon(file(paste0(idat_dir, "GSE85210_series_matrix.txt.gz"), "rb"))
lines <- readLines(con); close(con)
for (line in lines) {
    if (startsWith(line, "!Sample_geo_accession"))
        samples <- strsplit(gsub('"', '', line), '\t')[[1]][-1]
    if (startsWith(line, "!Sample_characteristics_ch1") && grepl("subject status", line))
        labels <- strsplit(gsub('"', '', line), '\t')[[1]][-1]
}

idat_files <- list.files(idat_dir, pattern="_Grn.idat.gz$")
basenames <- sub("_Grn.idat.gz$", "", idat_files)
gsm_ids <- sapply(strsplit(basenames, "_"), `[`, 1)
basename_df <- data.frame(sample=gsm_ids, basename=basenames, stringsAsFactors=FALSE)
df <- data.frame(sample=samples, label=labels, stringsAsFactors=FALSE)
samplesheet <- merge(basename_df, df, by="sample")
samplesheet$smoking <- ifelse(grepl("non", samplesheet$label), 0, 1)
samplesheet$idat_dir <- idat_dir
cat("Samplesheet rows:", nrow(samplesheet), "\n")

norm_methods <- c("illumina", "funnorm", "quantile")
filter_snps_opts <- c(TRUE, FALSE)
filter_sex_opts  <- c(TRUE, FALSE)
total <- 12; pipeline_count <- 0
start_time <- proc.time()

for (norm in norm_methods) {
    for (snp in filter_snps_opts) {
        for (sex in filter_sex_opts) {
            pipeline_count <- pipeline_count + 1
            pid <- paste0("norm=", norm, "_snp=", snp, "_sex=", sex)
            outfile <- file.path(output_dir, paste0(pid, ".csv.gz"))
            if (file.exists(outfile)) {
                cat(sprintf("(%d/12) SKIP: %s\n", pipeline_count, pid)); next
            }
            cat(sprintf("(%d/12) RUNNING: %s\n", pipeline_count, pid))
            targets <- samplesheet
            targets$Basename <- file.path(targets$idat_dir, targets$basename)
            rgset <- read.metharray(targets$Basename, extended=TRUE, verbose=FALSE)
            if (norm == "illumina") mset <- preprocessIllumina(rgset)
            else if (norm == "funnorm") mset <- preprocessFunnorm(rgset)
            else mset <- preprocessQuantile(rgset)
            rm(rgset); gc()
            if (snp) mset <- dropLociWithSnps(mset)
            if (sex) {
                ann <- getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)
                sex_probes <- ann$Name[ann$chr %in% c("chrX", "chrY")]
                mset <- mset[!rownames(mset) %in% sex_probes,]
            }
            beta <- getBeta(mset)
            rm(mset); gc()
            write.csv(beta, gzfile(outfile))
            rm(beta); gc()
            elapsed <- (proc.time() - start_time)["elapsed"]
            avg <- elapsed / pipeline_count
            eta <- avg * (total - pipeline_count)
            cat(sprintf("   Done | ETA: %.1f min remaining\n", eta/60))
        }
    }
}
cat(sprintf("All 12 pipelines complete. Total: %.1f min\n", (proc.time()-start_time)["elapsed"]/60))
