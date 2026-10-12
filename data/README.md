# Data: liked-video histories

Four personal YouTube accounts (u1, u2, u3, u7), exported through the Java
client (`client/`) from the account holders' own liked-videos playlists.
Private and deleted entries were removed at export. Only u2 and u3 pass the
inclusion rules of the paper (Section 3.2.1); u1 and u7 are released for
completeness.

Each `user_videos_uN.json` is a flat array of objects:

```json
{"video_id": "...", "title": "...", "category_id": "...", "category": "...",
 "liked_at": "YYYY-MM-DD", "duration_seconds": 0, "weight": 0}
```

`weight` is what the client computed under its historical `ORIGINAL` scheme at
export time. No analysis reads it: every script recomputes the weight from
`duration_seconds` under the scheme being tested (`code/pipeline_common.py`).

## Consent and privacy

All four histories were exported by, or with the explicit consent of, the
individual account holders, who were informed of the purpose of the study and
of what would be published (see the Informed Consent Statement of the paper).
No account holder is identified by name.

## Terms of use

Released under CC BY 4.0 for the reproduction and extension of the study. Do not
attempt to identify the account holders.
