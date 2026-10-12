# External replication on public watch histories

The pre-specified protocol, with its record of deviations, is
[`PROTOCOL_external_replication.md`](PROTOCOL_external_replication.md).

## What is in this folder

| File | Content |
|---|---|
| `select_users.py`, `selected_users.txt` | seeded selection of the 82 users that fit one day of API quota |
| `fetch_video_metadata.py` | category and duration of each video from the YouTube Data API |
| `adapt_watch_dataset.py` | watch log + metadata -> per-user JSON in the schema of `data/` |
| `external_user_level.py` | user-level analysis of a `chrono_split.py` result (logs 14, 15, 17) |
| `external_data_quality.py` | data-quality summary (log 16) |
| `results/` | `chrono_split.py` output on the replication users: primary analysis and sensitivity analyses S1-S3 |
| `synthetic/` | the two synthetic populations of the code check (log 17), see `synthetic/README.md` |

`results/*.json` are the files that `chrono_split.py --out` wrote during the run of
2026-10-09; only `config.users` and `config.out` were shortened from absolute
paths of the machine that ran them to relative names. They carry per-user
metrics and no video metadata. `run_all.sh` reproduces logs 14, 15, lines 2-5 of
log 16 and Figure 2 from them.

## Regenerating from the raw data

The watch log is the public dataset of Lall et al. (2020), *A YouTube Dataset with
User-Level Usage Data* (Kaggle), file `youtube_watch_log.csv`. The video metadata
are not redistributed and must be fetched again (YouTube API terms, see below).
Steps 2-4 take three days at the default API quota.

1. Select the users (seeded, independent of the data values):
   ```bash
   python3 select_users.py --log youtube_watch_log.csv --out-log log_selected.csv --out-ids selected_users.txt
   ```
2. Fetch the metadata with your own API key; repeat daily until it prints `all done`:
   ```bash
   export YT_API_KEY=...        # Google Cloud project with "YouTube Data API v3" enabled
   python3 fetch_video_metadata.py --log log_selected.csv --out meta.jsonl
   python3 fetch_video_metadata.py --log log_selected.csv --out meta.jsonl --export meta.csv
   ```
   Videos deleted or made private since 2018 no longer return metadata, so a later
   fetch can return slightly fewer videos than the 388,979 of the original run.
3. Per-user files (logs 12 and 12b):
   ```bash
   python3 adapt_watch_dataset.py --events log_selected.csv --meta meta.csv \
           --out-dir users > ../logs/12_external_adapt.txt
   python3 adapt_watch_dataset.py --events log_selected.csv --meta meta.csv \
           --out-dir users_allwatches --keep-all-watches > ../logs/12b_external_adapt_allwatches.txt
   ```
4. Analysis (from the repository root): `./run_all.sh --watch-users external/users`
   regenerates logs 13, 14 and 18 in full from the new files. Log 16, line 1 included:
   ```bash
   cd external
   python3 external_data_quality.py --log log_selected.csv --meta meta.csv \
           --cs ../outputs/chrono_split_external.json > ../logs/16_external_data_quality.txt
   ```
   With `--by-year` the same command also prints the share of watch events without
   metadata per year (Section 3.2.3); its output is `logs/16b_external_missing_by_year.txt`.

Sensitivity analyses (protocol, Section 5): S1 runs `chrono_split.py` on
`users_allwatches/`, after giving each watch a unique id (`video_id#k`), because
`chrono_split.load()` deduplicates by `video_id`; S2 adds `--weight-scheme uniform`;
S3 adds `--max-age-days 360`. Each result goes through `external_user_level.py`.

## YouTube API policy constraints (Developer Policies, checked 2026-10-09)

- III.D.1: one API client = exactly one API project. Do not split the fetch across
  several projects or accounts to get more quota; at the default quota the job
  simply takes about three days.
- III.E.4: non-authorized API data may not be stored for more than 30 calendar
  days without refresh. Keep `meta.jsonl`, `meta.csv` and `users/` out of any
  public repository and delete them when the analysis is finished.
