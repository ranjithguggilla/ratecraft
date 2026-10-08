# data

Nothing in here is committed except this note.

`python scripts/run_all.py` (or `python -c "from pathlib import Path; from ratecraft.data import fetch; fetch(Path('data/raw'))"`)
downloads two files into `data/raw/`:

| file | rows | sha256 |
|---|---|---|
| `freMTPL2freq.rda` | 677,991 policies | `82c85985...bdfe15a3` |
| `freMTPL2sev.rda` | 26,444 claims | `78e6e501...e09b14b3` |

They come from the CASdatasets R package at commit
`227fb56b8734bdb7c0327a41180e01d2ddaeaf26` of
<https://github.com/dutangc/CASdatasets> (release 1.2-1). The fetch is a sparse,
blobless `git fetch` of just those two files, with a plain HTTPS download as the
fallback, and the digests in `ratecraft.config.SOURCE` decide whether a file is
accepted.

The package is licensed GPL (>= 2), which is why the data is downloaded on demand
instead of living in this MIT-licensed repository. The parsed tables are cached
as parquet in `data/interim/` together with the fitted stage models.
