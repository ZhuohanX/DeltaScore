#!/usr/bin/env bash
set -euo pipefail

# Original archival mirrors for auxiliary perturbation resources.
# These links may no longer return raw files. This script refuses to overwrite
# local resources when a mirror returns an HTML page instead of the expected
# data file.

download_checked() {
  local url="$1"
  local out="$2"
  local tmp="${out}.tmp"

  echo "Downloading ${out}"
  curl -L -fS "$url" -o "$tmp"

  if head -c 512 "$tmp" | grep -Eqi '<!doctype html|<html'; then
    rm -f "$tmp"
    echo "Refusing to save ${out}: mirror returned HTML instead of raw data." >&2
    return 1
  fi

  mv "$tmp" "$out"
}

download_checked 'https://cloud.tsinghua.edu.cn/f/a75689d0420c43b59fe5/?dl=1' cause_relation.txt
download_checked 'https://cloud.tsinghua.edu.cn/f/e469ab88bc1a4bd7b937/?dl=1' cause_vocab.txt
download_checked 'https://cloud.tsinghua.edu.cn/f/5cc13b8ffcef4d20a726/?dl=1' conceptnet_triple.csv
download_checked 'https://cloud.tsinghua.edu.cn/f/0a91ce4a764a4e2fb752/?dl=1' kg.txt
download_checked 'https://cloud.tsinghua.edu.cn/f/c2f513db651949d6a0ca/?dl=1' negation_prefix_vocab.txt
download_checked 'https://cloud.tsinghua.edu.cn/f/7d09cf693e8145f7b459/?dl=1' negation_vocab.txt
download_checked 'https://cloud.tsinghua.edu.cn/f/6bc765f5a0fc4d90b333/?dl=1' pronoun_relation.txt
download_checked 'https://cloud.tsinghua.edu.cn/f/10c3f2805a8b4740b9ff/?dl=1' pronoun_vocab.txt
download_checked 'https://cloud.tsinghua.edu.cn/f/749a8f5ef5da414d91c4/?dl=1' time_relation.txt
download_checked 'https://cloud.tsinghua.edu.cn/f/eb5b792370254fb9b0dd/?dl=1' time_vocab.txt
download_checked 'https://cloud.tsinghua.edu.cn/f/329b4d5d07824b36979f/?dl=1' word2kg.txt
