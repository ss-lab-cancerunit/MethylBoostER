library(data.table)
library(GenomicRanges)

# This pre-processing script combines EPIC data with 450k data (TCGA and Chopra)


# ------------------------------- First, read in the data:-----------------------------------
# read in EPIC data
# please note that this raw EPIC data is not publicly released yet.
EPIC_data <- readRDS('newdata/Percentagemethpersample_EPIC_NvallT_10x_UNITE25_328_samples_2020.10.26.rds')
EPIC_GRanges <- GRanges(seqnames =EPIC_data$chr, strand = EPIC_data$strand, ranges = IRanges(start = EPIC_data$start, end = EPIC_data$end))
mcols(EPIC_GRanges) <- EPIC_data[, 5:ncol(EPIC_data)] # all the samples

# read in Chopra data (from https://osf.io/usxrn/?show=revision)
load('Chopra_internal/beta.training.rda')
beta.training <- as.data.frame(beta.training)
beta.training$probe = row.names(beta.training)

# read in TCGA
TCGA_data <- fread('TCGA/TCGA_beta_values_filtered.csv')

# load 450k probe annotations (locations on the genome)
# This version is harmonised to hg38, so we can work in hg38 (EPIC is in hg38)
probe_location_data <- fread('jhu-usc.edu_BLCA.HumanMethylation450.24.lvl-3.TCGA-XF-AAMW-01A-11D-A42F-05.gdc_hg38.txt')
probe_location_data <- probe_location_data[, c('Composite Element REF', 'Chromosome', 'Start', 'End')]


# ------------------------------- Merge Chopra and TCGA:-----------------------------------
# get chopra GRanges
chopera_data <- merge(probe_location_data, beta.training, by.x = 'Composite Element REF', by.y = 'probe', all = TRUE, sort = FALSE)
chopera_GRanges <- GRanges(seqnames =chopera_data$Chromosome, strand ='*', ranges = IRanges(start = chopera_data$Start, end = chopera_data$End))
mcols(chopera_GRanges) <- chopera_data[, 5:ncol(chopera_data)] # all the samples

# Only keep chopra samples that are not old TCGA
load('Chopra_internal/pdata.training.rda') # this tells us which samples are TCGA
library(stringr)
if (mean(colnames(chopera_data[,5:length(chopera_data)]) == row.names(pdata.training)) == 1) { # the following code only works if columns are in same order as annotation
  Chopras <- !str_detect(pdata.training$Barcode, 'TCGA') # Chopra, not TCGA, samples
  
  only_chopras <- as.data.frame(chopera_data[,5:length(chopera_data)])
  only_chopras <- only_chopras[, Chopras]
  only_chopras <- cbind(chopera_data[, 1:4], only_chopras) # add the annotation columns
} else {
  print('need to rearrange mcols!')
}

# merge chopra and new TCGA samples by probe (keeps all probes)
chopra_and_TCGA <- merge(only_chopras, TCGA_data, by.x = 'Composite Element REF', by.y = 'V1', all.x = TRUE, all.y = TRUE, sort = FALSE)


# ------------------------------- Remove blacklisted probes and X chrom probes:-----------------------------
# exclude probes in two blacklists we found
probes_to_keep <- readRDS('blacklist/Array450k_removedSNPsPriceandNaeem_2papers_HG19.rds')
# yes it is HG19 but the probe ids are the same, so its fine if we just take these
chopra_and_TCGA <- chopra_and_TCGA[chopra_and_TCGA$`Composite Element REF` %in% probes_to_keep$Probe_ID, ]

# and also remove X chromosome probes
chopra_and_TCGA <- chopra_and_TCGA[chopra_and_TCGA$Chromosome != 'chrX', ]


# ------------------------------- Combine 450ks with EPIC :-----------------------------------
# put in GRanges
chopera_and_TCGA_GRanges <- GRanges(seqnames =chopra_and_TCGA$Chromosome, strand ='*', ranges = IRanges(start = chopra_and_TCGA$Start, end = chopra_and_TCGA$End))
mcols(chopera_and_TCGA_GRanges) <- chopra_and_TCGA[, 5:ncol(chopra_and_TCGA)] # all the samples

# Now to combine Chopra and EPIC
maxgap <- 50
seqlevelsStyle(chopera_and_TCGA_GRanges) <- seqlevelsStyle(EPIC_GRanges) # changing chromosome name syntax to the island one so they are using the same naming syntax

# find overlapping probes
overlaps <- findOverlaps(chopera_and_TCGA_GRanges, EPIC_GRanges, ignore.strand = TRUE, maxgap = maxgap)

# put samples together for the overlapping probes
df1 <- as.data.frame(chopera_and_TCGA_GRanges[queryHits(overlaps), ])
df2 <- as.data.frame(EPIC_GRanges[subjectHits(overlaps), ])
library(dplyr)
combined <- bind_cols(df1, df2)

# removing duplicates (where chopra maps to more than one EPIC probe), keeping chopra probes
library(stringr)
sample_cols <- colnames(combined)[str_starts(colnames(combined), '[X,T]')] # starts with X or T
# using this code: https://stackoverflow.com/a/30511758
combined_no_dups <- setDT(combined)[, lapply(.SD, mean, na.rm = TRUE), by = c('seqnames', 'start'), .SDcols = sample_cols] # this is missing the end and strand (and the metadata from the EPIC rows), but this is ok
combined_no_dups <- as.data.frame(combined_no_dups)

print(paste('Max gap is ', maxgap))
print('This is the dim of combined no dups, before adding to EPIC:')
print(dim(combined_no_dups))

# remove old large objects we won't need anymore
rm(EPIC_data)
rm(EPIC_GRanges)
rm(df2)
rm(combined)

# ------------------------------- Make the diagnoses labels:-----------------------------------
# first, let's do the EPIC diagnoses
EPIC_info <- fread('newdata/multiregionMatrix_201024.csv')

# changing ids into format of our columns:
transformed <- data.frame(id = EPIC_info$newsample.id, overallpath = EPIC_info$overallpath)
transformed$id <- str_replace(str_replace(transformed$id, '\\(', '.'), '\\)', '.') # remove brackets
transformed$id <- paste0('X', transformed$id) # add X at start
transformed <- distinct(transformed) # remove duplicate row names problem
row.names(transformed) <- transformed$id

mapping <- data.frame(colname = colnames(combined_no_dups))
mapping$overallpath <- transformed[colnames(combined_no_dups), 'overallpath']

# two duplicate colnames need to be dealt with:
mapping[mapping$colname == 'X0474.T.1', 'overallpath'] <- 1
mapping[mapping$colname == 'X6029.T6.1', 'overallpath'] <- 1

label_mapping <- list('normal' = 0, 'ccRCC' = 1, 'chRCC' = 2, 'onc' = 3, 'prccT1' = 4, 'pRCCT2' = 5)

# next, chopra diagnoses
# from the chopera example code (website linked above) we need pdata for the chopera diagnoses
load('Chopra_internal/pdata.training.rda')
pdata_only_chopras <- pdata.training[Chopras, ] # Chopras from way above when we read in only_chopras
chopera_data_info <- data.frame(label = pdata_only_chopras$TissueType, id = paste0('X', row.names(pdata_only_chopras)))
chopera_data_info$label <- as.character(chopera_data_info$label)
mapping <- left_join(mapping, chopera_data_info, by = c('colname' = 'id'))

# now to add TCGA diagnoses
TCGA_diagnoses <- fread('TCGA/TCGA_diganoses.csv')
colnames(combined_no_dups)[str_detect(colnames(combined_no_dups), 'TCGA')][1:5]
colnames(TCGA_data)[2:length(TCGA_data)][1:5]
mean(str_replace_all(colnames(TCGA_data)[2:length(TCGA_data)], '-', '.') == colnames(combined_no_dups)[str_detect(colnames(combined_no_dups), 'TCGA')]) # are the TCGA data columns in the same order as the merged TCGA columns? Yes
mean(str_replace_all(colnames(TCGA_data)[2:length(TCGA_data)], '-', '.') == mapping$colname[str_detect(mapping$colname, 'TCGA')])
mapping$label[str_detect(mapping$colname, 'TCGA')] <- TCGA_diagnoses$x

# ------------------------------- Remove AMLs and EPIC replicates :-----------------------------------
# Next, we want to remove AMLs (from chopra data)
to_remove <- mapping[!is.na(mapping$label) & mapping$label == 'AML', 'colname']
combined_no_dups <- combined_no_dups[, !(colnames(combined_no_dups) %in% to_remove)]
mapping <- mapping[!mapping$colname %in% to_remove, ]

label_mapping <- list('Normal' = 0, 'KIRC' = 1, 'KICH' = 2, 'oncocytoma' = 3, 'KIRP' = 4, 'normal' = 0)

# turn labels in mapping to overallpath (ie, a number between 0 and 4)
for (r in 1:nrow(mapping)) {
  if (!is.na(mapping[r, 'label'])) {
    mapping[r, 'overallpath'] <- label_mapping[[as.character(mapping[r, 'label'])]]
  }
}

# remove three EPIC replicates
# one is .r, the other two are the duplicate colnames we dealt with above:
mapping[str_detect(mapping$colname, '.r'), ]

if (sum(is.na(combined_no_dups$X6262.T1.z)) < sum(is.na(combined_no_dups$X6262.T1.z.r))) {
  combined_no_dups$X6262.T1.z.r <- NULL # removing sample
  mapping <- mapping[mapping$colname != 'X6262.T1.z.r', ]
} else {
  print('need to alter code and delete other replicate!')
}
if (sum(is.na(combined_no_dups$X0474.T)) < sum(is.na(combined_no_dups$X0474.T.1))) {
  combined_no_dups$X0474.T.1 <- NULL
  mapping <- mapping[mapping$colname != 'X0474.T.1', ]
} else {
  print('need to alter code and delete other replicate!')
}
if (sum(is.na(combined_no_dups$X6029.T6)) < sum(is.na(combined_no_dups$X6029.T6.1))) {
  combined_no_dups$X6029.T6.1 <- NULL
  mapping <- mapping[mapping$colname != 'X6029.T6.1', ]
} else {
  print('need to alter code and delete other replicate!')
}

# and remove sample 0227.T1 as Sabrina said it may be a mix up
combined_no_dups$X0227.T1 <- NULL
mapping <- mapping[mapping$colname != 'X0227.T1', ]
# and remove its normal sample, as a normal sample without a corresponding tumour sample will mess up my get_folds() logic later on (it assumes each EPIC normal sample has at least one tumour sample)
combined_no_dups$X0227.N1 <- NULL
mapping <- mapping[mapping$colname != 'X0227.N1', ]

# 4 EPIC oncocytoma samples have a high CHG methylation, which means they are dodgy samples, so we remove them here
# samples to remove: 0218.T1, 0286.T1, 0330.N1, 0330.T1
combined_no_dups$X0218.T1 <- NULL
mapping <- mapping[mapping$colname != 'X0218.T1', ]

combined_no_dups$X0286.T1 <- NULL
mapping <- mapping[mapping$colname != 'X0286.T1', ]

combined_no_dups$X0330.N1 <- NULL
mapping <- mapping[mapping$colname != 'X0330.N1', ]

combined_no_dups$X0330.T1 <- NULL
mapping <- mapping[mapping$colname != 'X0330.T1', ]



# ------------------------------- Remove probes that are NA from one source:--------------------------------
# Next, remove probes that are NA for one whole source
mapping$source <- ifelse(str_detect(mapping$colname, 'TCGA'), 'TCGA', ifelse(str_detect(mapping$colname, 'R'), 'Chopra', ifelse(str_detect(mapping$colname, 'X'), 'EPIC', '')))
mapping[mapping$source == 'Chopra', 'colname']
chopra_data_processed <- combined_no_dups[, mapping[mapping$source == 'Chopra', 'colname']]
TCGA_data_processed <- combined_no_dups[, mapping[mapping$source == 'TCGA', 'colname']]
EPIC_data_processed <- combined_no_dups[, mapping[mapping$source == 'EPIC', 'colname']]
# chopra and EPIC don't have any whole NA probes (these prints are here to notify us in case this changes)
if (sum(rowMeans(is.na(chopra_data_processed)) == 1) != 0) {
  print('Need to add probes to be removed!')
}
if (sum(rowMeans(is.na(EPIC_data_processed)) == 1) != 0) {
  print('Need to add probes to be removed!')
}
# TCGA has about 5000 to remove:
probes_to_remove <- rowMeans(is.na(TCGA_data_processed)) == 1

combined_no_dups <- combined_no_dups[!probes_to_remove, ]


mean(mapping$colname == colnames(combined_no_dups)) # should be 1

# ------------------------------- Finally, save processed data and diagnoses:--------------------------------
write.table(combined_no_dups, paste0('training_testing_maxgap_', maxgap, '_newdata.csv'), quote = FALSE, sep = ',')
write.table(mapping[, c('colname', 'overallpath')], paste0('training_testing_diagnoses_maxgap_', maxgap, '_newdata.csv'), quote = FALSE, sep = ',')



