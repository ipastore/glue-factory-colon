# Experiment Matrix — All Pipeline Combinations

**Milestone:** Finish all Experiments
**Labels:** `experiment`, `P1-high`

Track all extractor+matcher and end-to-end combinations. Check off each one when the experiment issue is created and linked. Add the EXP-ID and link to the results comment when done.

---

## End-to-End

| # | Model | EXP-ID |
|---|---|---|---|
| | ROMA | | |
| | Master | | |

---

## SIFT Extractors

### py_colmap

| # | Matcher | Variant | EXP-ID |
|---|---|---|---|---|
| | NN | — | | |
| | LG (SIFT) | `oob` | | |
| | LG (SIFT) | `ft_ENDO_HOMO` | | |
| | LG (SIFT) | `ft_ENDO_3D` | | |
| | ROMA | — | | |

### py_cudasift

| # | Matcher | Variant | EXP-ID |
|---|---|---|---|---|
| | NN | — | | |
| | LG (SIFT) | `oob` | | |
| | LG (SIFT) | `ft_ENDO_HOMO` | | |
| | LG (SIFT) | `ft_ENDO_3D` | | |
| | ROMA | — | | |

---

## SuperPoint Extractors

### SuperPoint `oob`

| # | Matcher | Variant | EXP-ID |
|---|---|---|---|---|
| | NN | — | | |
| | LG (SuperPoint) | `oob` | | |
| | LG (SuperPoint) | `ft_ENDO_HOMO` | | |
| | LG (SuperPoint) | `ft_ENDO_3D` | | |
| | ROMA | — | | |

### SuperPoint `esuperpoint`

| # | Matcher | Variant | EXP-ID |
|---|---|---|---|---|
| | NN | — | | |
| | LG (SuperPoint) | `oob` | | |
| | LG (SuperPoint) | `ft_ENDO_HOMO` | | |
| | LG (SuperPoint) | `ft_ENDO_3D` | | |
| | ROMA | — | | |

### SuperPoint `ours`

| # | Matcher | Variant | EXP-ID |
|---|---|---|---|---|
| | NN | — | | |
| | LG (SuperPoint) | `oob` | | |
| | LG (SuperPoint) | `ft_ENDO_HOMO` | | |
| | LG (SuperPoint) | `ft_ENDO_3D` | | |
| | ROMA | — | | |

---

<!--
Fill in the EXP-ID column when you create the sub-issue.
-->