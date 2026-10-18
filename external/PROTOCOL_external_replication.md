# Pre-specified protocol: external replication on a public watch-history dataset

Status: written 2026-10-09, before any analysis of the dataset and before any video metadata were fetched. The archive had been opened beforehand only to read its README and column header and to count events per user (see Deviations). The first version-control commit containing this file is dated 2026-10-12.
To keep its meaning, commit/timestamp this file (git commit, OSF or Zenodo) before running anything on the data.
Anything decided or changed after seeing the data goes to the section "Deviations" at the bottom, with date and reason.

Dataset: S. Lall, M. Agarwal, R. Sivakumar, "A YouTube dataset with user-level usage data: Baseline characteristics
and key insights", ICC 2020 (Kaggle: youtube_watch_log.csv; 1,844,919 watch events, 244 users, 1,172,110 videos,
2010-08-08 to 2018-09-26; columns log_id, user_id, video_id, watch_date, subscribed, playlist_name).
The file carries no category and no duration. Both are fetched by video_id from the YouTube Data API v3
(fetch_video_metadata.py), i.e. as they are at fetch time, not at watch time. Deleted/private videos return
nothing and are dropped; their share is reported overall and per user.
Licence / terms: to be copied from the Kaggle page into this file before use: ______________________

## 1. Question and hypothesis
Does exponential decay of event weights (lambda = 0.95, half-life 13.5 d) improve prediction of the category of
future watched videos over the no-decay control (lambda = 1.0, W = infinity)?
H1 (primary): per-user mean recall@3 difference (0.95 minus 1.0) is positive.
This is the same comparison, same metric and same lambda as the confirmatory comparison on the like data
(Section 3.4 of the manuscript). Nothing is tuned on the new data.

## 2. Unit of analysis
The user. Each user's differences are first averaged over that user's origins (equal weight per origin, as in the
manuscript); users are then the independent units. (On the like data the unit was the origin because n = 2.)

## 3. Event construction (fixed now)
- Event = first watch of a video by a user (dedup by video_id, keep earliest date). Reason: same convention as
  the like data (a video is one event). Sensitivity analysis S1: all watches kept (repeat watches are events).
- Date = calendar date of watch_date as given in the file (time zone not documented; used as is). Duration, category = API metadata (value at collection time).
- Category: dataset category -> the 17 registered categories by exact name; categories with no registered
  counterpart are dropped. The mapping table is saved with the results; every dropped category and its count is
  reported. No ad-hoc merging after seeing results.
- Videos with missing category or duration are dropped; the share dropped is reported per user and overall.
- Weight scheme: continuous_log (as in the manuscript).


## 3a. User subset (fixed 2026-10-09, before any metadata was fetched)
One day of the default YouTube API quota covers ~475,000 unique videos, far fewer than the 1,172,110 in the log.
Therefore a random subset of users is analysed: `select_users.py`, seed 0, users shuffled with numpy
default_rng(0) and added in that order until their unique videos would exceed 475,000.
Result of the rule on the full log: 82 of 244 users, 473,125 unique videos (~9,463 API calls);
ids in `selected_users.txt`. The other 162 users are not used; the reason (API quota) and this rule are stated
in the paper. The subset is not changed after seeing any result. If budget permits later, the remaining users
may be added as a separately reported extension, never merged silently.

## 4. User inclusion (same thresholds as the manuscript, applied identically to all users)
>= 20 distinct dates, first origin at the 30th event, >= 30 training and >= 10 test events per origin,
horizon 60 days (disjoint windows), >= 3 retained origins. Number of users excluded by each rule is reported.
If fewer than 30 users qualify, results are reported as descriptive only.

## 5. Analysis
Primary: mean over users of the recall@3 difference, with a percentile bootstrap over users (B = 10,000, seed 0)
95% interval; exact two-sided sign test over users with non-zero difference; Wilcoxon signed-rank as a check.
Secondary (reported regardless of outcome): acc@1, MRR, nDCG@5 (Holm-adjusted across the four metrics);
the full lambda grid; spread of the user-level effect (SD, share positive); effect by history length tertile.
Sensitivity: S1 (all watches), S2 (weight scheme uniform), S3 (W = 360 days control at the same truncation).

## 6. Interpretation rules (fixed now)
- Replication: primary interval excludes zero on the positive side, direction as on the like data.
- Non-replication: interval includes zero or lies below zero. Reported as such; no change to the claim on likes,
  but the paper then states that the effect on likes was not reproduced on watches.
- In either case both parts are reported in the paper; the like data are never dropped because of the outcome here.
- Watch history is an implicit signal shaped by YouTube's own recommender; results are not described as
  "interests" or "likes".

## Deviations (append only)
- 2026-10-09: the archive was opened to read README.txt and the CSV header and to count events, distinct
  dates and spans per user (inputs to the inclusion rules only). No category, duration or outcome was
  available or computed. Finding: no category/duration columns -> metadata step via the API added above.
  Inclusion rules and analysis plan unchanged.
- 2026-10-09: metadata fetched (82 users, 473,125 videos; 388,979 with metadata, 84,146 unavailable = 17.8%;
  268 videos in unregistered categories ids 30/43/44). Analysis run the same day exactly as specified.
- Implementation note for S1 (all watches): chrono_split.load() deduplicates by video_id, so for S1 each watch
  was given a unique id (video_id#k) before running; no other change.
- Inclusion: 82 users -> 76 analysed (2 with < 20 distinct dates, 4 with < 3 retained origins); 1,637 origins.
