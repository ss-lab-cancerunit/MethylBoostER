library(edgeR)
library(DESeq2)
library(stringr)


# for this you will need an rda file of the expression data from the KIRC TCGA project
load(paste0('/Tank/methylation-patterns-code/methylation-patterns-izzy/data_preprocessing/KIRCexpression.rda'))
count_KIRC <- assay(data)
coldata_KIRC <- colData(data)

condition <- coldata_KIRC$definition != 'Solid Tissue Normal' # true if cancer present
mislabelled <- c("TCGA-B0-4699", "TCGA-B0-4696","TCGA-B0-4821","TCGA-B0-5083","TCGA-B0-5117","TCGA-B0-4688","TCGA-AK-3440","TCGA-AK-3433")
mislabelled_dont_have <- c('TCGA-AS-3777','TCGA-BP-4994','TCGA-A3-3374','TCGA-AK-3465','TCGA-AK-3447','TCGA-B2-3923', 'TCGA-BP-4334')

is_mislabelled <- ifelse(coldata_KIRC$patient %in% mislabelled, 1, ifelse(coldata_KIRC$patient %in% mislabelled_dont_have, 2, 0))

patient_id <- str_sub(coldata_KIRC$patient, 9, 12)

dge <- DGEList(counts=count_KIRC, group=condition)
#Filter out lowly expressed genes
keep <- rowSums(cpm(dge)>1) >= 2
dge <- dge[keep, , keep.lib.sizes=FALSE]
groupLevels <- relevel(dge$samples$group, ref="TRUE")
group <- factor(groupLevels)
coldata <- data.frame(condition=groupLevels, rownames=colnames(dge))
dds <- DESeqDataSetFromMatrix(countData = dge,
                              colData = coldata,
                              design= ~ condition)

vsd <- vst(dds,blind=TRUE)

library(ggplot2)
library(ggrepel)

# lets plot a UMAP of the expression
library(umap)
embedding <- umap(t(as.matrix(assay(vsd)))) 

to_plot <- data.frame(embedding$layout)
mean(row.names(to_plot) == coldata_KIRC$barcode) # should be 1
to_plot$is_mislabelled <- as.factor(is_mislabelled)
to_plot$condition <- condition
to_plot$patient_id <- patient_id

pdf('UMAPs/UMAP_KIRC_expression_data_mislabelled_coloured.pdf')
ggplot(to_plot) + 
  geom_point(aes(x=X1, y=X2, color=condition), alpha=0.4) +
  geom_point(data = to_plot[to_plot$is_mislabelled == 2, ], aes(x = X1, y = X2), shape=21, stroke=0.5, size=2, colour='orange') +
  geom_point(data = to_plot[to_plot$is_mislabelled == 1, ], aes(x = X1, y = X2), shape=21, stroke=0.5, size=2, colour='red') +
  geom_text_repel(data = to_plot[to_plot$is_mislabelled == 1, ], aes(x = X1, y = X2, label=patient_id), size=5, min.segment.length=0.03, force_pull = 0.5, force = 10) +
  scale_color_manual(values=c("#56B4E9", "#007d5b")) +
  theme_bw() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), text = element_text(size=14)) +
  xlab('UMAP 1') + 
  ylab('UMAP 2')
dev.off()


      