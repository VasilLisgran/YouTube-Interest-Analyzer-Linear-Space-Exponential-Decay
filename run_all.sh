#!/usr/bin/env bash
# Regenerate the raw console logs in logs/ from the data in data/.
#
#   ./run_all.sh            offline steps only (seconds to a few minutes)
#   ./run_all.sh --labse    also steps 01-02 (need the LaBSE download; 20-60+ min)
#   ./run_all.sh --check    write to outputs/logs_check/ and compare with logs/
#                           instead of overwriting them (exit code 1 on any difference)
#
# The scripts run with outputs/ as the working directory. outputs/ is a sibling
# of code/ and data/, so the '../data/...' paths that the scripts print into the
# logs are valid, and the logs come out byte for byte as committed.
set -euo pipefail
cd "$(dirname "$0")"

LABSE=0; CHECK=0
for arg in "$@"; do
  case "$arg" in
    --labse) LABSE=1 ;;
    --check) CHECK=1 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

PY=${PYTHON:-python3}
mkdir -p outputs
if [ "$CHECK" = 1 ]; then
  mkdir -p outputs/logs_check/extra
  LOGS=../outputs/logs_check
  FIG=../outputs/figure1_check.png
else
  LOGS=../logs
  FIG=../figures/figure1.png
fi
cd outputs

U=(../data/user_videos_u1.json ../data/user_videos_u2.json \
   ../data/user_videos_u3.json ../data/user_videos_u7.json)
A=(../data/user_videos_u2.json ../data/user_videos_u3.json)   # the two accounts that qualify

step() { echo "[run_all] $*"; }

if [ "$LABSE" = 1 ]; then
  step "01 component audit (LaBSE)"
  $PY ../code/evaluate_v2.py --embedder labse --algorithm dbscan --eps 0.51 \
      --dedup-key title --seeds 20 --users "${A[@]}" > "$LOGS/01_evaluate.txt"
  step "02 clustering eps sweep, DBSCAN vs HDBSCAN (LaBSE)"
  $PY ../code/sweeps.py --mode clustering --embedder labse --verbose \
      --users "${A[@]}" > "$LOGS/02_clustering_sweep.txt"
fi

step "03 N_eff lambda sweep"
$PY ../code/sweeps.py --mode lambda --users "${U[@]}" \
    --reference-date 2026-08-26 > "$LOGS/03_lambda_sweep.txt"

step "04 weight-scheme diagnostic"
set +o pipefail   # 'head' closes the pipe early
for ws in original continuous_log uniform; do
  echo "### weight scheme: $ws ###"
  for u in 2 3; do
    echo "--- u$u / $ws ---"
    $PY ../code/decay_diagnostics.py --input ../data/user_videos_u$u.json \
        --reference-date 2026-08-26 --lambda 0.95 --max-days 360 \
        --weight-scheme "$ws" 2>/dev/null | head -26
  done
done > "$LOGS/04_weight_ablation.txt"
set -o pipefail

step "05 rolling-origin decay evaluation (main result) and Figure 1"
$PY ../code/chrono_split.py --users "${U[@]}" --out chrono_split.json \
    > "$LOGS/05_chrono_split.txt"
$PY ../code/make_figure1.py --input chrono_split.json --output "$FIG"

step "06 decay vs hard truncation window"
$PY ../code/window_control.py --users "${U[@]}" --block 2 --seeds 8 \
    > "$LOGS/06_window_control.txt"

step "07 weight scheme x truncation age"
# The scripts print 'written: <path>' into the log; /tmp is what the committed log shows.
for ws in original continuous_log uniform; do
  for W in 0 360; do
    echo "########## weight_scheme=$ws  W=$W ##########"
    $PY ../code/chrono_split.py --users "${U[@]}" --weight-scheme "$ws" \
        --max-age-days "$W" --out "/tmp/abl_${ws}_${W}.json"
    echo
  done
done > "$LOGS/07_weight_truncation_ablation.txt"

step "08 tie-breaking sensitivity"
$PY ../code/tie_sensitivity.py --users "${A[@]}" --seeds 40 --block 2 --focus 0.95 \
    > "$LOGS/08_tie_sensitivity.txt"

step "extra: stratified block bootstrap, sign-flip test, N_eff at W=inf"
$PY ../code/stratified_bootstrap_check.py > "$LOGS/extra/stratified_bootstrap.txt"
$PY ../code/sign_flip_test.py            > "$LOGS/extra/sign_flip.txt"
$PY ../code/neff_untruncated.py --users "${U[@]}" > "$LOGS/extra/neff_untruncated.txt"

if [ "$CHECK" = 1 ]; then
  echo
  status=0
  for f in "$LOGS"/*.txt "$LOGS"/extra/*.txt; do
    rel=${f#"$LOGS"/}
    if cmp -s "$f" "../logs/$rel"; then echo "OK    $rel"
    else echo "DIFF  $rel"; status=1; fi
  done
  [ "$LABSE" = 1 ] || echo "(steps 01-02 were skipped; use --labse to include them)"
  exit $status
fi
