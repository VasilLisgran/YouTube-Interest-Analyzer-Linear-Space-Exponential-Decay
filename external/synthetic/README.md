# Synthetic populations for the code check (log 17, Section 3.5 of the paper)

Two populations of 30 simulated users each, in the schema of `data/`:

- `drift/`: over 1,000 days from 2015-01-01, each user is active on about 60% of days
  with 1-3 events per active day; 70% of events fall in three favourite categories,
  which are redrawn every 90 days, and the other 30% are drawn at random from the
  same pool of 12 registered categories; durations are uniform between 30 and 1,800 seconds.
- `stable/`: the same, except that the favourite categories never change.

In the drifting population the recent past predicts the next window, so decay ought
to help; in the stable one the whole history is the better guide. The populations
test the evaluation code and are not evidence about any real user.

These files are the exact inputs of the committed log 17 and of the synthetic
rows of log 18; `run_all.sh` reproduces both from them. The generator script
used to create them was not kept, so the files themselves are released.
