#!/usr/bin/env bash

if (( ! $# )); then
  echo "Usage: $(basename $0) matrix_file [matrix_file ...]"
  echo "Example1: $(basename $0) *.i1.prm"
  echo "Example2: $(basename $0) *.mtx"
  exit 1
fi

PAGERANK=../release/apps/pagerank/pagerank

if [[ ! -f "${PAGERANK}" ]]; then
  echo "Please build release version of pagerank executable"
  exit 1
fi

read -rd '' -a MATRIX_FILES <<< "$@"

echo "file,size,matrix_read_time_ms,pagerank_time_ms,residual,iterations"
LOG="$(basename $0).log"
for matrix_file in "${MATRIX_FILES[@]}"; do
  if [[ "${matrix_file}" == *.mtx || "${matrix_file}" == *.i1.prm ]]; then
    size=$(wc -c "${matrix_file}" | awk '{print $1}')
    echo -n "${matrix_file},${size},"
    if [[ "${matrix_file}" == *.i1.prm ]]; then
      "${PAGERANK}" --matrix_file="${matrix_file}" --mmap_prm_file > "${LOG}"
    else
      "${PAGERANK}" --matrix_file="${matrix_file}" > "${LOG}"
    fi
    read_time_ms="$(awk '/^matrix_read_time_ms: / {print $2}' "${LOG}")"
    pagerank_time_ms="$(awk '/^pagerank_time_ms: / {print $2}' "${LOG}")"
    residual="$(awk '/^residual: / {print $2}' "${LOG}")"
    iters="$(awk '/^iterations: / {print $2}' "${LOG}")"
    echo "${read_time_ms},${pagerank_time_ms},${residual},${iters}"
  fi
done
rm -rf "${LOG}"