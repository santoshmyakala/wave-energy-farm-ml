# Data

**Wave Energy Converters** dataset — UCI Machine Learning Repository (Neshat et al.), licensed **CC BY 4.0**.

Each row is one layout of 16 fully-submerged three-tether buoys in a wave farm at one of four real Australian coastal sites, simulated with each site's wave climate:

| Columns | Meaning |
|---|---|
| `X1..X16`, `Y1..Y16` | Buoy positions in metres (0–566 m; buoys are at least 50 m apart) |
| `P1..P16` | Power absorbed by each buoy (W) |
| `Powerall` | Total farm power (W) |

The files have no header row. `data/sample/` holds the **first 1,500 rows of each site's file**, for tests and CI. The full files (about 130 MB) are not committed. Download them from the UCI repository into `data/raw/`:

```
data/raw/Adelaide_Data.csv  data/raw/Perth_Data.csv  data/raw/Sydney_Data.csv  data/raw/Tasmania_Data.csv
```

## Data-quality findings (from `wec.data.clean`)

| Site | Rows | Exact duplicates removed | Rows where `Powerall` ≠ ΣPᵢ (>0.01%) removed | Rows used |
|---|---:|---:|---:|---:|
| Adelaide | 71,999 | 0 | 7 | 71,992 |
| Perth | 72,000 | 242 | 0 | 71,758 |
| Sydney | 72,000 | **27,174** | 7 | 44,819 |
| Tasmania | 72,000 | 0 | 11 | 71,989 |

Removing duplicates before splitting matters most for Sydney. Without it, more than a third of that site's rows would appear in both the training and test sets, which inflates the test scores.
