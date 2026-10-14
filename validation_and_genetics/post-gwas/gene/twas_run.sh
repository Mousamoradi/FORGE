#!/bin/bash
# S-PrediXcan (15 traits x 18 tissues x eQTL/sQTL) and S-MultiXcan, MetaXcan v0.7.5 with GTEx v8 MASHR models.
set -euo pipefail
W=$OUT/twas; mkdir -p $W/spx/eqtl $W/spx/sqtl $W/smx
TISSUES=$(cat $FORGE/gene/tissues.txt)
GW="--keep_non_rsid --snp_column SNP --effect_allele_column effect_allele --non_effect_allele_column non_effect_allele --beta_column beta --pvalue_column pvalue"

for q in eqtl sqtl; do
  for i in $(seq 1 15); do
    for tis in $TISSUES; do
      python $METAXCAN/SPrediXcan.py --model_db_path $MASHR_DIR/$q/mashr_$tis.db --model_db_snp_key varID \
        --covariance $MASHR_DIR/$q/mashr_$tis.txt.gz --gwas_file $W/gwas/fcm_z$i.txt.gz $GW \
        --additional_output --output_file $W/spx/$q/fcm_z${i}__$tis.csv
    done
  done
done

for q in eqtl sqtl; do
  L=expression; [ $q = sqtl ] && L=splicing
  for i in $(seq 1 15); do
    python $METAXCAN/SMulTiXcan.py --models_folder $MASHR_DIR/$q --models_name_pattern "mashr_(.*).db" \
      --model_db_snp_key varID --snp_covariance $SMX_COV/gtex_v8_${L}_mashr_snp_smultixcan_covariance.txt.gz \
      --metaxcan_folder $W/spx/$q --metaxcan_filter "fcm_z${i}__(.*).csv" --metaxcan_file_name_parse_pattern "(.*)__(.*).csv" \
      --gwas_file $W/gwas/fcm_z$i.txt.gz $GW --cutoff_condition_number 30 --output $W/smx/fcm_z${i}_$q.txt
  done
done
