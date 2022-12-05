#!/usr/bin/env bash

if (( ! $# )); then
  echo "Usage: $(basename $0) matrix_file [matrix_file ...]"
  echo "Example1: $(basename $0) \"\$(ls *.i0.prm)\""
  echo "Example2: $(basename $0) \"\$(ls *.mtx)\""
  exit 1
fi

CONNECTED_COMPONENTS=../release/apps/connected_components/connected_components

if [[ ! -f "${CONNECTED_COMPONENTS}" ]]; then
  echo "Please build release version of connected_components executable"
  exit 1
fi

IFS=$'\n' read -rd '' -a MATRIX_FILES <<< "$@"

echo -n "file,size,matrix_read_time_ms,connected_component_time_ms,"
echo "connected_components,iterations,converged"
LOG="$(basename $0).log"
for matrix_file in "${MATRIX_FILES[@]}"; do
  echo -n "${matrix_file},"
  size=$(wc -c "${matrix_file}" | awk '{print $1}')
  "${CONNECTED_COMPONENTS}" --matrix_file="${matrix_file}" > "${LOG}"
  read_time_ms="$(awk '/^matrix_read_time_ms: / {print $2}' "${LOG}")"
  cc_time_ms="$(awk '/^connected_component_time_ms: / {print $2}' "${LOG}")"
  cc_num="$(awk '/^connected_components: / {print $2}' "${LOG}")"
  iters="$(awk '/^iterations: / {print $2}' "${LOG}")"
  converged="$(awk '/^converged: / {print $2}' "${LOG}")"
  echo "${size},${read_time_ms},${cc_time_ms},${cc_num},${iters},${converged}"
done
rm -rf "${LOG}"