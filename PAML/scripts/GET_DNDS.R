#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(biomaRt)
  library(dplyr)
  library(readr)
  library(stringr)
})

# ------------------ EDIT ------------------
# Input: one gene per line OR a TSV with a column "gene" or "GeneSymbol"
IN_FILE <- "top500.txt"   # e.g., top500 genes as symbols OR Ensembl IDs
OUT_FILE <- "human_vs_macaque_dnds.tsv"
SPECIES_PREFIX <- "mmulatta"    # Macaca mulatta prefix in Ensembl homology attributes
ORTHOLOGY_KEEP <- c("ortholog_one2one")  # adjust if you want one2many etc.
# ------------------------------------------

read_gene_list <- function(path) {
  x <- read_lines(path, skip_empty_rows = TRUE)
  x <- x[!str_detect(x, "^#")]
  if (length(x) == 0) stop("No genes read from input file.")
  x
}

# Heuristic: Ensembl gene IDs look like ENSG...
is_ensembl_gene_id <- function(x) str_detect(x, "^ENSG\\d+")

message("Connecting to Ensembl BioMart...")
human <- useEnsembl(biomart = "ensembl", dataset = "hsapiens_gene_ensembl")

# Discover the correct attribute names for macaque DN/DS in *your* current Ensembl
attrs <- listAttributes(human)

find_attr <- function(pattern) {
  hit <- attrs$name[str_detect(attrs$name, pattern)]
  if (length(hit) == 0) return(NA_character_)
  hit[[1]]
}

# Common Ensembl attribute naming pattern in the human mart:
#   mmulatta_homolog_dn, mmulatta_homolog_ds, mmulatta_homolog_orthology_type, ...
attr_orth_type <- find_attr(paste0("^", SPECIES_PREFIX, "_homolog_orthology_type$"))
attr_dn        <- find_attr(paste0("^", SPECIES_PREFIX, "_homolog_dn$"))
attr_ds        <- find_attr(paste0("^", SPECIES_PREFIX, "_homolog_ds$"))
attr_dnds      <- find_attr(paste0("^", SPECIES_PREFIX, "_homolog_dn_ds$|^", SPECIES_PREFIX, "_homolog_dnds$"))

attr_target_gene <- find_attr(paste0("^", SPECIES_PREFIX, "_homolog_ensembl_gene$"))
attr_target_prot <- find_attr(paste0("^", SPECIES_PREFIX, "_homolog_ensembl_peptide$|^", SPECIES_PREFIX, "_homolog_ensembl_peptide_id$"))

needed <- c("ensembl_gene_id", "hgnc_symbol",
            attr_target_gene, attr_target_prot,
            attr_orth_type, attr_dn, attr_ds, attr_dnds)

if (any(is.na(needed))) {
  message("Could not find one or more macaque homology attributes in this Ensembl mart.")
  message("Missing: ", paste(needed[is.na(needed)], collapse=", "))
  message("Tip: inspect listAttributes(human) and search for 'mmulatta_homolog_' manually.")
  stop("Attribute discovery failed.")
}

genes <- read_gene_list(IN_FILE)

# Step 1: map input -> ensembl_gene_id (if needed)
if (all(is_ensembl_gene_id(genes))) {
  map_df <- tibble(input = genes, ensembl_gene_id = genes)
} else {
  message("Mapping HGNC symbols -> Ensembl gene IDs...")
  map_df <- getBM(
    attributes = c("hgnc_symbol", "ensembl_gene_id"),
    filters    = "hgnc_symbol",
    values     = unique(genes),
    mart       = human
  ) %>%
    as_tibble() %>%
    rename(input = hgnc_symbol)

  # Keep only the symbols requested, preserve input order
  map_df <- tibble(input = genes) %>%
    left_join(map_df, by = "input")
}

message("Mapped ", sum(!is.na(map_df$ensembl_gene_id)), " / ", nrow(map_df), " inputs to Ensembl IDs.")

# Step 2: query homology dN/dS vs macaque
message("Querying macaque homology dN/dS via BioMart...")

res <- getBM(
  attributes = needed,
  filters    = "ensembl_gene_id",
  values     = unique(na.omit(map_df$ensembl_gene_id)),
  mart       = human
) %>%
  as_tibble()

# Normalize column names
colnames(res) <- c(
  "ensembl_gene_id", "hgnc_symbol",
  "macaque_ensembl_gene", "macaque_ensembl_protein",
  "orthology_type", "dn", "ds", "dn_ds"
)

# Join back to original input order
out <- map_df %>%
  left_join(res, by = "ensembl_gene_id") %>%
  mutate(
    dn = suppressWarnings(as.numeric(dn)),
    ds = suppressWarnings(as.numeric(ds)),
    dn_ds = suppressWarnings(as.numeric(dn_ds)),
    dn_over_ds_calc = ifelse(!is.na(dn) & !is.na(ds) & ds > 0, dn / ds, NA_real_)
  )

# Optional: filter to one2one orthologs
out_filt <- out %>%
  filter(is.na(orthology_type) | orthology_type %in% ORTHOLOGY_KEEP)

# Write
write_tsv(out_filt, OUT_FILE, na = "NA")

message("Wrote: ", OUT_FILE)
message("Rows: ", nrow(out_filt))
message("Non-missing dn/ds (from Ensembl field): ", sum(!is.na(out_filt$dn_ds)))
message("Non-missing dn/ds (computed dn/ds where ds>0): ", sum(!is.na(out_filt$dn_over_ds_calc)))