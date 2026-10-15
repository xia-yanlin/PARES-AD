args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1 || length(args) > 2) {
  stop("Expected a temporary-directory argument and optional values mode")
}

root <- args[[1]]
values_mode <- length(args) == 2 && identical(args[[2]], "values")

load(file.path(root, "UCBERKELEY_AMY_6MM.rda"))
amyloid <- UCBERKELEY_AMY_6MM[, c("RID", "SCANDATE", "qc_flag", "TRACER", "AMYLOID_STATUS")]
amyloid[["anchor_available"]] <- !is.na(amyloid[["AMYLOID_STATUS"]])
if (values_mode) {
  write.csv(amyloid, file.path(root, "amy_values.csv"), row.names = FALSE)
}
amyloid[["AMYLOID_STATUS"]] <- NULL
write.csv(amyloid, file.path(root, "amy_meta.csv"), row.names = FALSE)

load(file.path(root, "UCBERKELEY_TAU_6MM.rda"))
tau <- UCBERKELEY_TAU_6MM[, c("RID", "SCANDATE", "qc_flag", "TRACER", "META_TEMPORAL_SUVR")]
tau_value <- suppressWarnings(as.numeric(tau[["META_TEMPORAL_SUVR"]]))
tau[["anchor_available"]] <- is.finite(tau_value) & tau_value > 0
if (values_mode) {
  write.csv(tau, file.path(root, "tau_values.csv"), row.names = FALSE)
}
tau[["META_TEMPORAL_SUVR"]] <- NULL
write.csv(tau, file.path(root, "tau_meta.csv"), row.names = FALSE)

load(file.path(root, "PACC.rda"))
pacc <- PACC[, c("RID", "PACC_VISDATE", "mPACCtrailsB")]
pacc[["anchor_available"]] <- !is.na(pacc[["mPACCtrailsB"]])
if (values_mode) {
  write.csv(pacc, file.path(root, "pacc_values.csv"), row.names = FALSE)
}
pacc[["mPACCtrailsB"]] <- NULL
write.csv(pacc, file.path(root, "pacc_meta.csv"), row.names = FALSE)
