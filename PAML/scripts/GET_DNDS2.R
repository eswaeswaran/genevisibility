#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(biomaRt)
  library(dplyr)
  library(readr)
  library(stringr)
  library(httr)
  library(jsonlite)
  library(purrr)
  library(tibble)
})

IN_FILE <- "top500.txt"
OUT_FILE <- "human_vs_macaque_dnds.tsv"

TARGET_SPECIES_REST <- "macaca_mulatta"   # REST API species name
SPECIES_PREFIX_MART <- "mmulatta"         # BioMart prefix
ORTHOLOGY_KEEP <- c("ortholog_one2one")

read_gene_list <- function(path) {
  x <- read_lines(path)
  x <- x[!str_detect(x, "^\\s*$")]
  x <- x[!str_detect(x, "^#")]
  if (length(x) == 0) stop("No genes read from input file.")
  x
}

is_ensembl_gene_id <- function(x) str_detect(x, "^ENSG\\d+")

message("Connecting to Ensembl BioMart...")
human <- useEnsembl(biomart = "genes", dataset = "hsapiens_gene_ensembl", mirror = "useast")

attrs <- listAttributes(human)

find_attr <- function(pattern) {
  hit <- attrs$name[str_detect(attrs$name, pattern)]
  if (length(hit) == 0) return(NA_character_)
  hit[[1]]
}

# We ONLY require attributes that BioMart still provides
attr_orth_type   <- find_attr(paste0("^", SPECIES_PREFIX_MART, "_homolog_orthology_type$"))
attr_target_gene <- find_attr(paste0("^", SPECIES_PREFIX_MART, "_homolog_ensembl_gene$"))

needed_mart <- c("ensembl_gene_id", "hgnc_symbol", attr_target_gene, attr_orth_type)
if (any(is.na(needed_mart))) {
  message("Could not find required macaque orthology attributes in this Ensembl mart.")
  message("Missing: ", paste(needed_mart[is.na(needed_mart)], collapse=", "))
  message("Try searching listAttributes(human)$name for 'mmulatta_homolog_'")
  stop("BioMart attribute discovery failed.")
}

genes <- read_gene_list(IN_FILE)

# Step 1: map input -> ensembl_gene_id (if needed)
if (all(is_ensembl_gene_id(genes))) {
  map_df <- tibble(input = genes, ensembl_gene_id = genes)
} else {
  message("Mapping HGNC symbols -> Ensembl gene IDs...")
  map_df0 <- getBM(
    attributes = c("hgnc_symbol", "ensembl_gene_id"),
    filters    = "hgnc_symbol",
    values     = unique(genes),
    mart       = human
  ) %>% as_tibble() %>% rename(input = hgnc_symbol)

  map_df <- tibble(input = genes) %>%
    left_join(map_df0, by = "input")
}

message("Mapped ", sum(!is.na(map_df$ensembl_gene_id)), " / ", nrow(map_df), " inputs to Ensembl IDs.")

# Step 2: query orthology IDs via BioMart (no dn/ds here)
message("Querying macaque orthology (gene IDs + orthology type) via BioMart...")
res_mart <- getBM(
  attributes = needed_mart,
  filters    = "ensembl_gene_id",
  values     = unique(na.omit(map_df$ensembl_gene_id)),
  mart       = human
) %>% as_tibble()

colnames(res_mart) <- c("ensembl_gene_id", "hgnc_symbol", "macaque_ensembl_gene", "orthology_type")

# Optional: keep only desired orthology types
res_mart <- res_mart %>%
  filter(is.na(orthology_type) | orthology_type %in% ORTHOLOGY_KEEP)

# Step 3: fetch dn/ds/dnds via Ensembl REST (when available)
rest_base <- "https://rest.ensembl.org"

fetch_dnds_rest <- function(human_ensembl_gene_id) {
  url <- paste0(
    rest_base,
    "/homology/id/human/", human_ensembl_gene_id,
    "?target_species=", TARGET_SPECIES_REST,
    ";type=orthologues;sequence=none;format=full"
  )

  r <- GET(url, add_headers(`Content-Type` = "application/json"))
  if (status_code(r) == 429) {
    # rate limit: back off and retry once
    Sys.sleep(1.5)
    r <- GET(url, add_headers(`Content-Type` = "application/json"))
  }
  if (status_code(r) >= 400) {
    return(tibble(
      ensembl_gene_id = human_ensembl_gene_id,
      dn = NA_real_, ds = NA_real_, dnds = NA_real_
    ))
  }

  x <- content(r, as = "text", encoding = "UTF-8")
  j <- fromJSON(x, simplifyVector = FALSE)

  homs <- j$data[[1]]$homologies
  if (is.null(homs) || length(homs) == 0) {
    return(tibble(
      ensembl_gene_id = human_ensembl_gene_id,
      dn = NA_real_, ds = NA_real_, dnds = NA_real_
    ))
  }

  # keep only rows where target species matches (should already be filtered)
  rows <- map_dfr(homs, function(h) {
    tibble(
      ensembl_gene_id = human_ensembl_gene_id,
      target_id = h$target$id,
      type = h$type,
      dn = suppressWarnings(as.numeric(h$dn)),
      ds = suppressWarnings(as.numeric(h$ds)),
      dnds = suppressWarnings(as.numeric(h$dnds))
    )
  })

  # If multiple orthologues exist, keep one2one first if present, else first row.
  rows %>%
    arrange(desc(type == "ortholog_one2one")) %>%
    slice(1) %>%
    select(ensembl_gene_id, dn, ds, dnds)
}

message("Fetching dn/ds/dnds via Ensembl REST (this may take a bit for 500 genes)...")
rest_df <- map_dfr(unique(na.omit(res_mart$ensembl_gene_id)), function(id) {
  Sys.sleep(0.05) # be polite to the API
  fetch_dnds_rest(id)
})

# Step 4: assemble final output in original input order
out <- map_df %>%
  left_join(res_mart, by = "ensembl_gene_id") %>%
  left_join(rest_df, by = "ensembl_gene_id") %>%
  mutate(
    dn_over_ds_calc = ifelse(!is.na(dn) & !is.na(ds) & ds > 0, dn / ds, NA_real_)
  )

write_tsv(out, OUT_FILE, na = "NA")

message("Wrote: ", OUT_FILE)
message("Rows: ", nrow(out))
message("Non-missing dnds (REST field): ", sum(!is.na(out$dnds)))
message("Non-missing dn/ds computed (ds>0): ", sum(!is.na(out$dn_over_ds_calc)))