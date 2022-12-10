#!/usr/bin/env bash

if (( ! $# )); then
  echo "Usage: $(basename $0) prm_file [prm_file ...]"
  echo "Example: $(basename $0) *.prm"
  exit 1
fi

PRM_PRINT=../release/apps/prm_print/prm_print

if [[ ! -f "${PRM_PRINT}" ]]; then
  echo "Please build release version of prm_print executable"
  exit 1
fi

read -rd '' -a PRM_FILES <<< "$@"

echo -n "file,size,rows,cols,nnz,symmetric,"
echo -n "idx_item_size,idx_shift,idx_min_val,idx_max_val,"
echo "pos_item_size,pos_shift,pos_min_val,pos_max_val"
LOG="$(basename $0).log"
for prm_file in "${PRM_FILES[@]}"; do
  if [[ "${prm_file}" == *.prm ]]; then
    size=$(wc -c "${prm_file}" | awk '{print $1}')
    echo -n "${prm_file},${size},"
    "${PRM_PRINT}" --prm_file="${prm_file}" > "${LOG}"
    rows="$(awk '/^rows: / {print $2}' "${LOG}")"
    cols="$(awk '/^cols: / {print $2}' "${LOG}")"
    nnz="$(awk '/^nnz: / {print $2}' "${LOG}")"
    symmetric="$(awk '/^symmetric: / {print $2}' "${LOG}")"
    item_sizes=($(awk '/^  item_size: / {print $2}' "${LOG}"))
    shifts=($(awk '/^  shift_by_min_val: / {print $2}' "${LOG}"))
    min_vals=($(awk '/^  min_val: / {print $2}' "${LOG}"))
    max_vals=($(awk '/^  max_val: / {print $2}' "${LOG}"))
    echo -n "${rows},${cols},${nnz},${symmetric},"
    echo -n "${item_sizes[0]},${shifts[0]},${min_vals[0]},${max_vals[0]},"
    echo "${item_sizes[1]},${shifts[1]},${min_vals[1]},${max_vals[1]}"
  fi
done
rm -rf "${LOG}"