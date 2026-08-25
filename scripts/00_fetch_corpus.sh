#!/usr/bin/env bash
# Fetch the four documentation corpora at pinned commits, then extract the
# real per-file last-modified time from each repo's git history.
set -e
fetch () {  # name url subdir commit
  local name=$1 url=$2 sub=$3 sha=$4
  [ -d "$name" ] && { echo "$name exists, skipping"; return; }
  git clone --filter=blob:none --no-checkout "$url" "$name"
  git -C "$name" sparse-checkout init --cone
  git -C "$name" sparse-checkout set "$sub"
  git -C "$name" checkout "$sha"
  echo "$name @ $(git -C "$name" rev-parse HEAD)"
}
fetch corpus-src    https://github.com/mdn/content.git              files/en-us  6cee0131a446a08d9664179233dc2cabc89fe065
fetch corpus-dotnet https://github.com/dotnet/docs.git              docs         414c78268538ed216b3aa3e25d22976c5830131d
fetch corpus-k8s    https://github.com/kubernetes/website.git       content/en   5836bf49ac2bb44f166741a2d9224d68986ee66e
fetch corpus-ha     https://github.com/home-assistant/home-assistant.io.git source ee175d3e4d1738984a654ed9f2dd2fa0f195dcc4

mkdir -p data
bash scripts/00_mtimes.sh corpus-src    files/en-us data/mtimes_mdn.json
bash scripts/00_mtimes.sh corpus-dotnet docs        data/mtimes_dotnet.json
bash scripts/00_mtimes.sh corpus-k8s    content/en  data/mtimes_k8s.json
bash scripts/00_mtimes.sh corpus-ha     source      data/mtimes_ha.json
