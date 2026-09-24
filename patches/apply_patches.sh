#!/bin/bash
# Apply documented modifications to the RecursiveMAS clone.
#
#   bash patches/apply_patches.sh
#
# Each patch is applied on a branch and written out as a .patch file so the
# exact diff can go into the thesis appendix. Backups (*.bak) are kept.
#
# PATCH 001 — re-expose --method
#   Upstream implements four collaboration methods (text, text_recursive,
#   ours, ours_recursive) but hardcodes args.method = "ours_recursive" at
#   inference_mas.py:1941 and defines no --method argument. Three fully
#   implemented paths are therefore unreachable. This patch re-exposes them,
#   which the text-mas baseline needs.
#
#   Justification for the thesis: the code paths are upstream and correspond to
#   baselines reported in the paper; the release merely pins the runner to one.
#   Re-exposing the switch lets every baseline run through the identical
#   evaluation harness, which is a precondition for comparability.

set -e

WS="${WS:-/pfs/work9/workspace/scratch/ma_jkliem-jkliem_ma}"
R="$WS/RecursiveMAS"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAS="$R/inference/inference_utils/inference_mas.py"
RUN="$R/inference/run.py"

[ -d "$R/.git" ] || { echo "not a git repo: $R"; exit 1; }

cd "$R"
git checkout -b thesis-patches 2>/dev/null || git checkout thesis-patches

if grep -q '"--method"' "$RUN"; then
  echo "patch 001 already applied"; exit 0
fi

cp "$MAS" "$MAS.bak"; cp "$RUN" "$RUN.bak"

python3 - "$MAS" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()

anchor = '    parser.add_argument("--solver_pre_question", type=int, default=0)'
if anchor not in s:
    sys.exit("anchor not found in inference_mas.py; inspect manually")
s = s.replace(anchor, anchor + '\n'
    '    parser.add_argument("--method", type=str, default="ours_recursive",\n'
    '                        choices=["text", "text_recursive", "ours", "ours_recursive"],\n'
    '                        help="Collaboration method. Re-exposed by thesis patch 001.")', 1)

old = '    args = parse_args()\n    args.method = "ours_recursive"'
if old not in s:
    sys.exit("hardcoded assignment not found; inspect manually")
s = s.replace(old, '    args = parse_args()\n'
                   '    # thesis patch 001: method now comes from --method\n', 1)
open(p, "w").write(s); print("  patched inference_mas.py")
PY

python3 - "$RUN" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()

a1 = '    p.add_argument("--temperature", type=float, default=0.6)'
if a1 not in s: sys.exit("temperature anchor not found in run.py")
s = s.replace(a1, a1 + '\n'
    '    p.add_argument("--method", type=str, default="ours_recursive",\n'
    '                   choices=["text", "text_recursive", "ours", "ours_recursive"],\n'
    '                   help="Collaboration method. Re-exposed by thesis patch 001.")', 1)

a2 = '    out.append("--do_sample")'
if a2 not in s: sys.exit("do_sample anchor not found in run.py")
s = s.replace(a2, '    out.extend(["--method", str(args.method)])\n' + a2, 1)
open(p, "w").write(s); print("  patched run.py")
PY

git diff > "$HERE/001-expose-method.patch"
echo
echo "patch written to patches/001-expose-method.patch"
echo "verify with:  cd $R && python inference/run.py --help | grep -A2 method"