#!/usr/bin/env bash

if (( ! $# )); then
  echo "Usage: $(basename $0) mtx_file [mtx_file ...]"
  echo "Example: $(basename $0) *.mtx"
  exit 1
fi

MTX_TO_PRM=../release/apps/mtx_to_prm/mtx_to_prm

if [[ ! -f "${MTX_TO_PRM}" ]]; then
  echo "Please build release version of mtx_to_prm executable"
  exit 1
fi

read -rd '' -a MTX_FILES <<< "$@"

echo -n "file,mtx_size,mtx_read_time_ms,i0_write_time_ms,i0_prm_size,"
echo "i1_write_time_ms,i1_prm_size"
LOG="mtx_to_prm.log"
for mtx_file in "${MTX_FILES[@]}"; do
  # Find "graph" .mtx files
  if head -50 "${mtx_file}" | grep '%%MatrixMarket matrix coordinate' \
    &> /dev/null; then
    mtx_size=$(wc -c "${mtx_file}" | awk '{print $1}')
    echo -n "${mtx_file},${mtx_size},"
    # Get name of the matrix without the .mtx suffix
    mtx="${mtx_file%.*}"
    "${MTX_TO_PRM}" --mtx_file="${mtx_file}" --output_row_major > "${LOG}"
    mtx_read_time_ms=$(awk '/^mtx_read_time_ms: / {print $2}' "${LOG}")
    i0_write_time_ms=$(awk '/^index_and_prm_write_time_ms: / {print $2}' \
      "${LOG}")
    i0_prm_size=$(wc -c "${mtx}.i0.prm" | awk '{print $1}')
    "${MTX_TO_PRM}" --mtx_file="${mtx_file}" --nooutput_row_major > "${LOG}"
    i1_write_time_ms=$(awk '/^prm_write_time_ms: / {print $2}' "${LOG}")
    i1_prm_size=$(wc -c "${mtx}.i1.prm" | awk '{print $1}')
    echo -n "${mtx_read_time_ms},${i0_write_time_ms},"
    echo "${i0_prm_size},${i1_write_time_ms},${i1_prm_size}"
  fi
done
rm -rf "${LOG}"