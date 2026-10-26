#!/usr/bin/env bash
# Optional public timing tool; dependencies must be installed separately.
set -euo pipefail
analysis_root="$(cd "$(dirname "$0")/../.." && pwd)"
tool_build="$analysis_root/rtl/build/analysis/tools"
mkdir -p "$tool_build"
fetch() {
  local url="$1" destination="$2" expected="$3"
  if [[ ! -f "$destination" ]]; then
    curl --fail --location --retry 3 --output "$destination" "$url"
  fi
  node --input-type=module - "$destination" "$expected" <<'JS'
import {readFileSync} from 'node:fs'
import {createHash} from 'node:crypto'
import {strict as assert} from 'node:assert'
import process from 'node:process'
assert.equal(createHash('sha256').update(readFileSync(process.argv[2])).digest('hex'),process.argv[3],'Pinned tool source checksum')
JS
}
fetch https://github.com/ivmai/cudd/archive/refs/tags/cudd-3.0.0.tar.gz "$tool_build/cudd.tgz" 5fe145041c594689e6e7cf4cd623d5f2b7c36261708be8c9a72aed72cf67acce
fetch https://github.com/The-OpenROAD-Project/OpenSTA/archive/150bc0dbd103cbc97c0f85f63c4c8338826121fd.tar.gz "$tool_build/opensta.tgz" 846b6e07fb7349be29cd005325aafe482fe8c78f03e200cdb76b194e0e31d02a
tar -xzf "$tool_build/cudd.tgz" -C "$tool_build"
tar -xzf "$tool_build/opensta.tgz" -C "$tool_build"
(
  cd "$tool_build/cudd-cudd-3.0.0"
  ./configure --prefix="$tool_build/cudd-install" --enable-silent-rules
  make -j "${BUILD_JOBS:-4}"
  make install
)
extra_args=()
if [[ "$(uname -s)" == Darwin ]]; then
  brew_prefix="$(brew --prefix)"
  extra_args+=("-DBISON_EXECUTABLE=$brew_prefix/opt/bison/bin/bison" "-DFLEX_EXECUTABLE=$brew_prefix/opt/flex/bin/flex" "-DFLEX_INCLUDE_DIR=$brew_prefix/opt/flex/include" "-DTCL_LIBRARY=$brew_prefix/opt/tcl-tk@8/lib/libtcl8.6.dylib" "-DTCL_HEADER=$brew_prefix/opt/tcl-tk@8/include/tcl.h" "-DCMAKE_PREFIX_PATH=$brew_prefix")
fi
cmake -S "$tool_build/OpenSTA-150bc0dbd103cbc97c0f85f63c4c8338826121fd" -B "$tool_build/opensta-build" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DCUDD_DIR="$tool_build/cudd-install" "${extra_args[@]}"
cmake --build "$tool_build/opensta-build" -j "${BUILD_JOBS:-4}"
"$tool_build/opensta-build/sta" -version
