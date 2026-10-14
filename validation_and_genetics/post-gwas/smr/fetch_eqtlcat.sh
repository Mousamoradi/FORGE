#!/bin/bash
# eQTL Catalogue GTEx v8 summary statistics (expression .all, leafcutter .cc), restricted by tabix to the fetch regions.
set -uo pipefail
W=$OUT/smr/eqtlcat; mkdir -p $W
cd $W
tail -n +2 $OUT/coloc/qtl/datasets.tsv | while IFS=$'\t' read -r did sg tis q n cs lbf path; do
  f=$W/$did.region.tsv.gz
  [ -s $f ] && continue
  url=${path/ftp:\/\/ftp.ebi.ac.uk/https://ftp.ebi.ac.uk}
  for k in 1 2 3; do
    tabix -h -R $OUT/smr/fetch_regions.tsv "$url" | bgzip -c > $f.part && gzip -t $f.part && mv $f.part $f && break
    sleep 30
  done
done
