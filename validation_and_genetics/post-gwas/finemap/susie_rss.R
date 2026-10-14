#!/usr/bin/env Rscript
# SuSiE-RSS for one region (L = 10, 95% coverage, fixed residual variance).
# args: <prefix>  reads <prefix>.dat.tsv and <prefix>.R.bin; writes .cs.txt, .pip.txt, .diag.txt, .fit.rds
suppressMessages({library(data.table); library(coloc); library(susieR)})
pref <- commandArgs(TRUE)[1]
d <- fread(paste0(pref, ".dat.tsv"))
p <- nrow(d)
R <- matrix(readBin(paste0(pref, ".R.bin"), "double", n = p * p), nrow = p)
R <- (R + t(R)) / 2; diag(R) <- 1

z <- d$BETA / d$SE
N <- as.integer(median(d$N))
s_rss <- tryCatch(estimate_s_rss(z = z, R = R, n = N), error = function(e) NA_real_)
ev <- min(eigen(R, symmetric = TRUE, only.values = TRUE)$values)
lam <- 0
if (ev < 1e-8) {
  lam <- -ev + 1e-4
  R <- (R + lam * diag(p)) / (1 + lam); diag(R) <- 1
}
dimnames(R) <- list(d$SNP, d$SNP)

D <- list(beta = d$BETA, varbeta = d$SE^2, snp = d$SNP, position = d$POS, type = "quant", N = N, sdY = 1,
          MAF = pmin(pmax(pmin(d$FREQ, 1 - d$FREQ), 1e-6), 0.5), LD = R)
S <- runsusie(D, L = 10, coverage = 0.95, estimate_residual_variance = FALSE,
              maxit = 1000, repeat_until_convergence = FALSE)
pip <- susie_get_pip(S)
write(pip, paste0(pref, ".pip.txt"), ncolumns = 1)
saveRDS(list(fit = S, snp = d$SNP, pos = d$POS, a1 = d$A1, a2 = d$A2, N = N, lambda = lam),
        paste0(pref, ".fit.rds"))

cs <- S$sets
lines <- character(0)
for (i in seq_along(cs$cs)) {
  idx <- as.integer(cs$cs[[i]])
  top <- idx[which.max(pip[idx])]
  lines <- c(lines, paste(names(cs$cs)[i], length(idx), round(cs$coverage[i], 4), top, round(pip[top], 6),
                          round(cs$purity[i, 1], 4), paste(idx, collapse = ","), sep = "\t"))
}
writeLines(lines, paste0(pref, ".cs.txt"))
writeLines(c(paste("s_rss", s_rss), paste("lambda", lam), paste("min_eig", ev)), paste0(pref, ".diag.txt"))
