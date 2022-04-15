# Multiple test correction from the multiple Fisher's tests we ran

p_vals <- c('turajlic'=0.00010658052305057398, 'HCMDB'=4.141725963571837e-17, 'dbEMT' = 1.0319735527032817e-16, 'cosmic'=4.676436768828098e-13, 'tfcheck'=3.583681364893711e-15, 'EpiFactor DB'=0.00985775997962868, 'KEGG_hsa05211_RCC'=0.11559247243935564, 'pRCC -harmonizome'=0.10300780961930461, 'RCC harmonizome'=5.093734462597253e-07)
p_vals

p.adjust(p_vals, "fdr")
