#!/usr/bin/env Rscript
# coloc.bf_bf of each region's GWAS SuSiE fit against eQTL Catalogue GTEx SuSiE LBF matrices.
# Both matrices are subset to the same variants in the same order, and QTL rows to worklist credible sets.
suppressMessages({library(data.table); library(coloc); library(parallel)})
O <- Sys.getenv("OUT"); Q <- file.path(O, "coloc", "qtl")
od <- file.path(O, "coloc", "gtex"); dir.create(od, recursive = TRUE, showWarnings = FALSE)
spec <- fread(file.path(O, "loci", "regions.tsv"))
work <- fread(file.path(Q, "worklist.tsv"))
dsets <- fread(file.path(Q, "datasets.tsv"))
vkey <- function(c, p, a, b) { a <- toupper(a); b <- toupper(b); paste0(c, ":", p, ":", pmin(a, b), "_", pmax(a, b)) }

GW <- list()
for (i in seq_len(nrow(spec))) {
  reg <- spec$region_id[i]
  f <- file.path(O, "finemap", "susie", paste0(reg, ".fit.rds"))
  if (!file.exists(f)) next
  G <- readRDS(f); S <- G$fit
  if (is.null(S$sets$cs_index) || !length(S$sets$cs_index)) next
  bf <- S$lbf_variable
  ix <- match(colnames(bf), G$snp)
  k <- vkey(spec$CHROM[i], G$pos[ix], G$a1[ix], G$a2[ix])
  colnames(bf) <- k
  bf <- bf[S$sets$cs_index, !duplicated(k), drop = FALSE]
  GW[[reg]] <- list(bf = bf, trait = spec$best_top_trait[i], ncs = length(S$sets$cs_index))
}

run <- function(did) {
  meta <- dsets[dataset_id == did]; WW <- work[dataset_id == did]
  L <- fread(file.path(Q, "lbf", paste0(did, ".lbf.tsv.gz")))
  sp <- tstrsplit(L$variant, "_", fixed = TRUE)
  L[, vk := vkey(sub("^chr", "", chromosome), position, sp[[3]], sp[[4]])]
  setkey(L, molecular_trait_id)
  lc <- grep("^lbf_variable", names(L), value = TRUE)
  out <- list()
  for (reg in intersect(unique(WW$region), names(GW))) {
    bf1 <- GW[[reg]]$bf
    for (mt in unique(WW[region == reg]$molecular_trait_id)) {
      w <- WW[region == reg & molecular_trait_id == mt]
      Lm <- L[.(mt), nomatch = 0L]
      if (!nrow(Lm)) next
      Mx <- t(as.matrix(Lm[, ..lc])); colnames(Mx) <- Lm$vk
      Ls <- as.integer(sub(".*_L", "", w$cs_id)); Ls <- Ls[Ls >= 1 & Ls <= nrow(Mx)]
      if (!length(Ls)) next
      bf2 <- Mx[Ls, !duplicated(colnames(Mx)), drop = FALSE]
      common <- intersect(colnames(bf1), colnames(bf2))
      if (length(common) < 500) next
      A <- bf1[, common, drop = FALSE]; B <- bf2[, common, drop = FALSE]
      stopifnot(identical(colnames(A), colnames(B)))
      r <- tryCatch(coloc.bf_bf(A, B, p1 = 1e-4, p2 = 1e-4, p12 = 5e-6, overlap.min = 0.5, trim_by_posterior = TRUE),
                    error = function(e) NULL)
      if (is.null(r) || is.null(r$summary) || !nrow(r$summary)) next
      s <- as.data.table(r$summary)
      s[, `:=`(region = reg, dataset_id = did, tissue = meta$sample_group, quant = meta$quant_method,
               molecular_trait_id = mt, gene_id = w$gene_id[1], trait = GW[[reg]]$trait, n_common = length(common))]
      out[[length(out) + 1]] <- s
    }
  }
  if (length(out)) fwrite(rbindlist(out, fill = TRUE), file.path(od, paste0(did, ".tsv.gz")))
  invisible(NULL)
}
invisible(mclapply(unique(work$dataset_id), run, mc.cores = as.integer(Sys.getenv("THREADS", "2"))))
