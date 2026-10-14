#!/usr/bin/env Rscript
# coloc.abf of each region's strongest trait against EyeGEx retina cis-eQTL (N = 406, df = 404).
suppressMessages({library(data.table); library(coloc)})
O <- Sys.getenv("OUT")
DF <- 404; NQ <- 406; MIN_SHARED <- 200
dir.create(file.path(O, "coloc", "retina"), recursive = TRUE, showWarnings = FALSE)
spec <- fread(file.path(O, "loci", "regions.tsv"))
vkey <- function(c, p, a, b) { a <- toupper(a); b <- toupper(b); paste0(c, ":", p, ":", pmin(a, b), "_", pmax(a, b)) }

for (i in seq_len(nrow(spec))) {
  reg <- spec$region_id[i]
  g <- fread(file.path(O, "finemap", "regions", paste0(reg, ".tsv")))
  g[, vk := vkey(CHR, POS, A1, A2)]
  g <- g[!duplicated(vk)]
  R <- fread(file.path(O, "coloc", "retina_regions", paste0("chr", spec$CHROM[i], ".tsv.gz")))[region == reg]
  R[, vk := vkey(chr, pos, effect, other)]
  R <- R[is.finite(p) & p > 0 & p < 0.9999 & is.finite(beta) & beta != 0]
  R[, se := abs(beta) / abs(qt(p / 2, df = DF, lower.tail = FALSE))]
  frac <- length(intersect(g$vk, unique(R$vk))) / nrow(g)
  out <- list()
  for (gg in unique(R$gene)) {
    Rg <- R[gene == gg][!duplicated(vk)]
    common <- intersect(g$vk, Rg$vk)
    if (length(common) < MIN_SHARED) next
    g1 <- g[match(common, vk)]; r1 <- Rg[match(common, vk)]
    bq <- ifelse(toupper(g1$A1) != toupper(r1$effect), -r1$beta, r1$beta)
    r <- coloc.abf(list(beta = g1$BETA, varbeta = g1$SE^2, snp = common, type = "quant",
                        N = as.integer(median(g1$N)), sdY = 1, MAF = pmin(pmax(pmin(g1$FREQ, 1 - g1$FREQ), 1e-6), 0.5)),
                   list(beta = bq, varbeta = r1$se^2, snp = common, type = "quant", N = NQ, sdY = 1),
                   p1 = 1e-4, p2 = 1e-4, p12 = 5e-6)
    res <- as.data.table(r$results); j <- which.max(res$SNP.PP.H4)
    s <- as.data.table(t(r$summary))
    s[, `:=`(region = reg, gene = gg, trait = spec$best_top_trait[i], n_common = length(common),
             frac_gwas_matched = frac, shared_vk = res$snp[j])]
    out[[length(out) + 1]] <- s
  }
  if (length(out)) fwrite(rbindlist(out), file.path(O, "coloc", "retina", paste0(reg, ".tsv.gz")))
  cat(reg, length(out), "genes; GWAS variant match", round(100 * frac, 1), "%\n")
}
