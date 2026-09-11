#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_file="${repo_root}/tools/local_seeded_engine/ExportSeeded.cpp"
output_file="${1:-${repo_root}/build/local_seeded_engine/libcg_seeded.so}"
compiler="${CXX:-g++}"

mkdir -p "$(dirname "${output_file}")"

"${compiler}" \
  -std=c++20 \
  -O3 \
  -DNDEBUG \
  -fPIC \
  -fvisibility=hidden \
  -static-libstdc++ \
  -static-libgcc \
  -Wl,--exclude-libs,ALL \
  -shared \
  "${source_file}" \
  -o "${output_file}"

echo "${output_file}"
