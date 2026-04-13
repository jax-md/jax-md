#!/usr/bin/env bash
#set -euo pipefail

# module load GROMACS/2025.3-foss-2025b-CUDA-12.9.1

OUTDIR="sp_alch"
TEMPLATE_MDP="prod_alch.mdp"
GRO="./mobley_3053621.gro"
TOP="./mobley_3053621.top"

mkdir -p "${OUTDIR}"

for window in $(seq 0 19); do
  window_tag=$(printf "window_%02d" "${window}")
  mdp_window="${OUTDIR}/prod_alch_${window_tag}.mdp"
  deffnm="${OUTDIR}/${window_tag}"
  edr="${deffnm}.edr"
  out="${OUTDIR}/sp_all_terms_${window_tag}.xvg"

  awk -v w="${window}" '
    BEGIN { done = 0 }
    /^[[:space:]]*init_lambda_state[[:space:]]*=/ {
      print "init_lambda_state        = " w
      done = 1
      next
    }
    { print }
    END {
      if (!done) {
        print "init_lambda_state        = " w
      }
    }
  ' "${TEMPLATE_MDP}" > "${mdp_window}"

  # 1) Build TPR from alchemical MDP for this lambda state
  gmx grompp \
    -f "${mdp_window}" \
    -c "${GRO}" \
    -p "${TOP}" \
    -o "${deffnm}.tpr"

  # 2) Single-point energy evaluation: rerun on one frame
  gmx mdrun \
    -deffnm "${deffnm}" \
    -rerun "${GRO}" \
    -nt 1

  # Read menu and get max index (handles lines with multiple index/name pairs)
  max_idx=$(
    gmx energy -f "${edr}" -o "/tmp/_ignore_${window_tag}.xvg" 2>&1 <<< "0" \
    | awk '
        {
          for (i=1; i<=NF; i++) {
            if ($i ~ /^[0-9]+$/ && $i+0 > max) max=$i+0
          }
        }
        END { print max }
      '
  )

  # Build "1 2 3 ... N"
  all_sel=$(seq 1 "${max_idx}" | paste -sd' ' -)

  # Extract all available terms for this window
  printf "%s\n0\n" "${all_sel}" | gmx energy -dp -f "${edr}" -o "${out}"
done

