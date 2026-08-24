#!/usr/bin/env bash
# Full Gemma-3-12B replication run set — mirrors the Qwen3.6-27B authoritative
# experiments in RESULTS.md. Everything below reuses the same run_* scripts
# pointed at configs/gemma3_12b.yaml; only the model + its lens differ.
#
# Runs on an ephemeral GPU box (A100-80GB fits 12B bf16 with room). Each step
# is save_run'd immediately, because .gitignore excludes data/results*/ etc. and
# the box is wiped on a timer — see scripts/save_run.py.
#
# PREREQ (do once, before running this):
#   1. google/gemma-3-12b-pt is GATED — accept the license on HF, then:
#        huggingface-cli login          # or: export HF_TOKEN=...
#   2. pip install -e '.[model]'         # heavy extras: torch/transformers/hf_hub
#
# Usage:
#   bash scripts/run_gemma3_12b_all.sh            # commit + push each step
#   NO_PUSH=1 bash scripts/run_gemma3_12b_all.sh  # commit only (no push creds)

set -euo pipefail

CFG=configs/gemma3_12b.yaml
SAVE_FLAGS=""
[ "${NO_PUSH:-0}" = "1" ] && SAVE_FLAGS="--no-push"
LOGDIR=runs/gemma3-12b/logs
mkdir -p "$LOGDIR"

# run <label> <logname> -- <command...>
#   tee the command's output to a log, then archive results under runs/<label>.
run () {
  local label="$1"; local logname="$2"; shift 3   # drop label, logname, "--"
  echo; echo "========== $label =========="; echo "+ $*"
  "$@" 2>&1 | tee "$LOGDIR/${logname}.log"
  python scripts/save_run.py "$label" --config "$CFG" $SAVE_FLAGS \
    --note "gemma3-12b: $logname"
}

# ---- 0. lens sanity: is the lens even reading on Gemma? (P0) ----------------
# If this diagonal isn't clean, STOP — every null below is otherwise attackable
# as "the lens doesn't work on Gemma". No save_run: gate prints, doesn't write
# to the config's results paths.
echo "========== replication_gate =========="
python scripts/replication_gate.py --config "$CFG" 2>&1 | tee "$LOGDIR/gate.log"

# ---- 1. fit role directions (needed by calibrate / diagnose / E3) -----------
run gemma-fit fit_directions -- \
  python scripts/fit_directions.py --config "$CFG" --export-pt

# ---- 2. RQ1 decodability — primary layer 36, both sites (P1) ----------------
run gemma-rq1-L36 rq1_L36 -- \
  python scripts/run_rq1.py --config "$CFG"

# ---- 3. RQ1 layer-band edges 18 and 44 (the config's own sweep) (P1) --------
run gemma-rq1-L18 rq1_L18 -- \
  python scripts/run_rq1.py --config "$CFG" --layer 18
run gemma-rq1-L44 rq1_L44 -- \
  python scripts/run_rq1.py --config "$CFG" --layer 44

# ---- 4. RQ1 second seed (only moves the random-subspace / label draw) (P2) --
run gemma-rq1-seed42 rq1_seed42 -- \
  python scripts/run_rq1.py --config "$CFG" --seed 42

# ---- 5. Within-pair decoding — the headline result, both sites (P0) ---------
run gemma-within-final within_final -- \
  python scripts/check_within_pair.py --config "$CFG" --site final_token
run gemma-within-entity within_entity -- \
  python scripts/check_within_pair.py --config "$CFG" --site entity_token

# ---- 6. RQ2 ablation, both sites (P1) ---------------------------------------
# NB: regenerates the verdict under the CURRENT logic (86c96cc), so the stale
# workspace_causally_involved=true label on the old pilot JSON is superseded.
run gemma-rq2 rq2 -- \
  python scripts/run_rq2.py --config "$CFG"

# ---- 7. E4 recruitment (P1) -------------------------------------------------
run gemma-e4 e4 -- \
  python scripts/run_e4.py --config "$CFG"

# ---- 8. Nonlinear probe — closes the multiplicative-code hatch (P1) ---------
run gemma-nonlinear nonlinear -- \
  python scripts/check_nonlinear_probe.py --config "$CFG"

# ---- 9. Ridge-penalty sweep — sets the wording discipline (P1) --------------
run gemma-ridge ridge -- \
  python scripts/check_ridge_penalty.py --config "$CFG"

# ---- 10. Push calibration + diagnosis (evidence push fails) (P0 for E3) -----
# Calibrate the identity-swap alpha and search the push grid. On Qwen this hit
# exit 3 (no grid value moved role behaviour); Gemma's pilot diagnose showed the
# same. Captured, not fatal.
set +e
python scripts/calibrate.py --config "$CFG" --site entity_token \
  --push-grid 0.5,1,2,4,8 2>&1 | tee "$LOGDIR/calibrate.log"
CAL_EXIT=$?
set -e
python scripts/diagnose_push.py --config "$CFG" --site entity_token \
  --coefficient 8 2>&1 | tee "$LOGDIR/diagnose.log"
python scripts/save_run.py gemma-calibrate --config "$CFG" $SAVE_FLAGS \
  --note "gemma3-12b: calibrate (exit $CAL_EXIT) + diagnose"

cat <<EOF

################################################################################
  Unattended run set complete. Everything above needed no human decision.

  ONE MANUAL STEP REMAINS — the primary / E3 addressability dissociation.
  run_primary reads model.push_coefficient and model.alpha from the CONFIG
  (not from calibration.json), and both are null in $CFG on purpose.

  1. Open data/gemma3_12b/calibration.json.
  2. Edit $CFG:
       model.alpha            <- the alpha calibrate wrote
       model.push_coefficient <- calibrate's value, OR (if it exited 3, as on
                                 Qwen) the largest grid value, 8, so E3 still
                                 produces the strength-check dissociation.
  3. Then run:
       python scripts/run_primary.py --config $CFG 2>&1 | tee $LOGDIR/primary.log
       python scripts/save_run.py gemma-primary --config $CFG $SAVE_FLAGS \\
         --note "gemma3-12b: E3 primary"

  NOTE: run_primary's verdict lives ONLY in the log (primary.log) — it prints to
  stdout and writes no summary JSON. Cite the log, per RESULTS.md.

  STILL NOT COVERED by this script (need a merged config, not a flag):
    - 6-pair RQ1: configs/expanded_pairs.yaml carries the QWEN model. Make a
      gemma+6pair config (gemma model block + expanded_pairs' concept_pairs)
      and rerun run_rq1.py against it.
################################################################################
EOF
