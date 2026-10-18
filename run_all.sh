#!/usr/bin/env bash
# Regenerate the raw console logs in logs/ and the two figures.
#
#   ./run_all.sh                 offline steps (about two minutes); overwrites logs/ and figures/
#   ./run_all.sh --check         same steps, written to outputs/check/ and compared with logs/
#                                (exit code 1 on any difference); logs/ is not touched
#   ./run_all.sh --labse         also steps 01-02 (LaBSE download; 20-60+ min on a laptop CPU)
#   ./run_all.sh --watch-users DIR
#                                also the steps that need the per-user watch files written by
#                                external/adapt_watch_dataset.py (logs 13 and 18 in full); the
#                                files are not distributed, see external/README_external.md
#
# Offline steps: 03-11 and 19-20 from data/; 14, 15 and the replication rows of Figure 2 from the
# committed replication results in external/results/; 17 and the synthetic rows of 18 from the
# synthetic populations in external/synthetic/; lines 2-5 of 16 from external/results/.
# Logs 12, 12b and line 1 of 16 need the raw watch log and the API metadata (external/README).
#
# The scripts run with outputs/ as the working directory, a sibling of code/ and data/, so the
# '../data/...' paths that they print inside the logs are the committed ones.
set -euo pipefail
cd "$(dirname "$0")"

LABSE=0; CHECK=0; WATCH=""
while [ $# -gt 0 ]; do
  case "$1" in
    --labse) LABSE=1 ;;
    --check) CHECK=1 ;;
    --watch-users) WATCH="$(cd "$2" && pwd)"; shift ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

PY=${PYTHON:-python3}
mkdir -p outputs
if [ "$CHECK" = 1 ]; then
  rm -rf outputs/check; mkdir -p outputs/check
  LOGS=../outputs/check; FIGS=../outputs/check
else
  LOGS=../logs; FIGS=../figures
fi
cd outputs

U=(../data/user_videos_u1.json ../data/user_videos_u2.json \
   ../data/user_videos_u3.json ../data/user_videos_u7.json)
A=(../data/user_videos_u2.json ../data/user_videos_u3.json)   # the two accounts that qualify
RES=../external/results
step() { echo "[run_all] $*"; }

# ---------------------------------------------------------------- LaBSE steps (optional)
if [ "$LABSE" = 1 ]; then
  step "01 component audit (LaBSE)"
  $PY ../code/evaluate_v2.py --embedder labse --algorithm dbscan --eps 0.51 \
      --dedup-key title --seeds 20 --users "${A[@]}" > "$LOGS/01_evaluate.txt"
  step "02 clustering eps sweep, DBSCAN vs HDBSCAN (LaBSE)"
  $PY ../code/sweeps.py --mode clustering --embedder labse --verbose \
      --users "${A[@]}" > "$LOGS/02_clustering_sweep.txt"
fi

# ---------------------------------------------------------------- like histories
step "03 N_eff lambda sweep"
$PY ../code/sweeps.py --mode lambda --users "${U[@]}" \
    --reference-date 2026-08-26 > "$LOGS/03_lambda_sweep.txt"

step "04 weight-scheme diagnostic"
for ws in original continuous_log uniform; do
  echo "### weight scheme: $ws ###"
  for u in 2 3; do
    echo "--- u$u / $ws ---"
    $PY ../code/decay_diagnostics.py --input ../data/user_videos_u$u.json \
        --reference-date 2026-08-26 --lambda 0.95 --max-days 360 --weight-scheme "$ws"
  done
done > "$LOGS/04_weight_ablation.txt"

step "05 rolling-origin decay evaluation (main result) and Figure 1"
$PY ../code/chrono_split.py --users "${U[@]}" --out chrono_split.json \
    > "$LOGS/05_chrono_split.txt"
$PY ../code/make_figure1.py --input chrono_split.json --output "$FIGS/figure1.png"

step "06 decay vs hard truncation window"
$PY ../code/window_control.py --users "${U[@]}" > "$LOGS/06_window_control.txt"

step "07 weight scheme x truncation age"
# chrono_split.py prints 'written: <path>'; the committed log was written with --out /tmp/x.json
for ws in original continuous_log uniform; do
  for W in 0 360; do
    echo "########## weight_scheme=$ws  W=$W ##########"
    $PY ../code/chrono_split.py --users "${U[@]}" --weight-scheme "$ws" \
        --max-age-days "$W" --out /tmp/x.json
    echo
  done
done > "$LOGS/07_weight_truncation_ablation.txt"

step "08 tie-break sensitivity of the confirmatory comparison"
$PY ../code/tie_sensitivity.py "${A[@]}" > "$LOGS/08_tie_sensitivity.txt"

step "09 multiplicity (Bonferroni, Holm)"
$PY ../code/multiplicity.py chrono_split.json > "$LOGS/09_multiplicity.txt"

step "10 candidate-set check"
$PY ../code/candidate_set_check.py "${A[@]}" > "$LOGS/10_candidate_set.txt"

step "11 data summary"
$PY ../code/data_summary.py "${U[@]}" > "$LOGS/11_data_summary.txt"

step "19 window control under 40 tie-break salts"
$PY ../code/window_tie_sensitivity.py "${U[@]}" > "$LOGS/19_window_tie_sensitivity.txt"

step "20 first-ranked category with and without decay"
$PY ../code/top_category_check.py "${U[@]}" > "$LOGS/20_top_category.txt"

# ---------------------------------------------------------------- watch histories
if [ -n "$WATCH" ]; then
  step "13 chrono_split.py on the watch histories in $WATCH"
  $PY ../code/chrono_split.py --users "$WATCH"/*.json --out chrono_split_external.json \
      > "$LOGS/13_external_chrono_split.txt"
  EXT=$PWD/chrono_split_external.json
else
  EXT=$(cd "$RES" && pwd)/chrono_split_external.json
fi

step "14 user-level analysis of the replication"
(cd ../external && $PY external_user_level.py "$EXT") > "$LOGS/14_external_user_level.txt"

step "21 user-level analysis without the duplicated history (user 213)"
(cd ../external && $PY external_user_level.py "$EXT" --drop-ids 213) > "$LOGS/21_external_user_level_drop213.txt"

step "15 pre-specified sensitivity analyses S1-S3"
for S in S1 S2 S3; do
  echo "########## sensitivity $S ##########"
  (cd ../external && $PY external_user_level.py "results/chrono_split_external_$S.json") > s.tmp
  head -n 6 s.tmp; rm -f s.tmp
  echo; echo
done > "$LOGS/15_external_sensitivity.txt"

step "16 data quality (lines 2-5; line 1 needs the metadata, see external/README_external.md)"
(cd ../external && $PY external_data_quality.py --cs "$EXT") > 16_lines_2_5.txt

step "17 code validation on synthetic populations"
OUT=$PWD
for pop in drift stable; do
  $PY ../code/chrono_split.py --users ../external/synthetic/$pop/*.json \
      --out chrono_split_synthetic_$pop.json > /dev/null
done
{
  echo "CODE VALIDATION ON SYNTHETIC DATA (known answer; not evidence about real users)"
  echo "drift: 30 users, 3 favourite categories (70% of events) redrawn every 90 days, 1000 days"
  echo "stable: same, favourites fixed"
  for pop in drift stable; do
    echo; echo "=== $pop ==="
    (cd ../external && $PY external_user_level.py "$OUT/chrono_split_synthetic_$pop.json")
  done
} > "$LOGS/17_code_validation_synthetic.txt"

step "18 drift diagnostic (exploratory)"
{
  $PY ../code/drift_diagnostic.py --label 'likes (u2,u3)' "${A[@]}"
  if [ -n "$WATCH" ]; then
    $PY ../code/drift_diagnostic.py --label 'watches (first watch)' "$WATCH"/*.json
  else
    grep '^watches' ../logs/18_drift_diagnostic.txt   # not recomputable without the watch files
  fi
  $PY ../code/drift_diagnostic.py --label 'synthetic drift' ../external/synthetic/drift/*.json
  $PY ../code/drift_diagnostic.py --label 'synthetic stable' ../external/synthetic/stable/*.json
} > 18.tmp
mv 18.tmp "$LOGS/18_drift_diagnostic.txt"

step "Figure 2"
$PY ../code/make_figure2.py --likes chrono_split.json --watches "$EXT" \
    --output "$FIGS/figure2.png" > /dev/null

# ---------------------------------------------------------------- comparison
if [ "$CHECK" = 1 ]; then
  echo
  status=0
  report() { if [ "$2" = 0 ]; then echo "OK    $1"; else echo "DIFF  $1"; status=1; fi; }
  for f in "$LOGS"/*.txt; do
    name=$(basename "$f")
    cmp -s "$f" "../logs/$name"; report "$name" $?
  done
  tail -n 4 ../logs/16_external_data_quality.txt | cmp -s - 16_lines_2_5.txt
  report "16_external_data_quality.txt (lines 2-5)" $?
  [ -n "$WATCH" ] || echo "      (18: the 'watches' line is copied from logs/, not recomputed; use --watch-users)"
  [ "$LABSE" = 1 ] || echo "      (01-02 skipped; use --labse)"
  echo "      (figures were regenerated into outputs/check/ and are not compared byte for byte)"
  exit $status
fi
