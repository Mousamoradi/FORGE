#!/bin/bash
# EyeGEx retina nominal cis-eQTL inside the regions. Positions from the GRCh38 columns (10);
# column 12 is the effect allele.
set -euo pipefail
W=$OUT/coloc/retina_regions; mkdir -p $W
SPEC=$OUT/loci/regions.tsv
for ch in $(awk -F'\t' 'NR>1{print $2}' $SPEC | sort -un); do
  gzip -dc $EYEGEX/new_nominal.eQTL.chr$ch.with_thresholds.txt.gz | awk -v CH=$ch -v RG=$SPEC '
    BEGIN{FS=OFS="\t"; while((getline l < RG)>0){split(l,a,"\t"); if(a[2]==CH){n++; s[n]=a[3]; e[n]=a[4]; r[n]=a[1]}}
          print "gene","region","chr","pos","effect","other","p","beta"}
    NR>1{p=$10+0; for(j=1;j<=n;j++) if(p>=s[j] && p<=e[j]){c=$9; sub(/^chr/,"",c); print $1,r[j],c,p,$12,$13,$14,$15}}' \
    | gzip > $W/chr$ch.tsv.gz
done
