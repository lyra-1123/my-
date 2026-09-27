#!/usr/bin/env bash
# 从 Google Drive「dukascopy导出」文件夹下载 XAUUSD M1 数据到 data/（约 360MB）。
# 前提：文件共享设置为"知道链接的任何人可查看"。若返回 HTML/403，请让用户在 Drive 里打开链接共享。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
FILES="
19NwZz07-Zx96opRH-h7RBUpJTyY2_PxS:2026
1gPIq_4ci7BfEa6HXdJDkegI3T5lm4iJH:2025
1uVPEx1n95KqeIfYKCvmPy_jB7GbrSzu4:2024
1_hC7L1siL9CW3bFaSaziFUFQoJvnmqUQ:2023
1HK7Nm8HIYunw0nPouxI7hBSe-OMKQ321:2022
1XqWxSv78oToJ39qBy_7vo-HVHByAIh7J:2021
11raTruC3LqQfGp0wTNAyZJXOqykC35P2:2020
1WEB6sBfI6LRnvUbd65HWG2vIQis8lYJR:2019
1wcG98M3CyaJ7UBj_Yy1ij_Ja6sUNUD7z:2018
1gcJb7dKp9aS6gQ06EzdiE94dJ1tSxgeB:2017
1i4RJ2NWur2tOKx7abNmHssz9xk9NATYf:2016_DUKAREAL
1mh5yzJKDZhQReA_HCUYFp-QU3dz0NHZG:2015_DUKAREAL
1p-2AuliHoiawjc-FRPUykWAJl2w2onV0:2014_DUKAREAL
1qI7XvG5NXkvwf-Tvr8Fr2IvwZVQFOmWM:2013_DUKAREAL
1X8KjnSlmQ1XiRN5WxWGDpFxqvAdp2tND:2012_DUKAREAL
13HYOTVUTRjDA77CTzwzi_sFAcs5cdKcP:2011_DUKAREAL
1QQQF-PgwRbSTFWsY4aUTzRGUGLAD1ze7:2010_DUKAREAL
1kWdC1R6N9LDjbP3eePtGMMNMKsJJou1a:2009_DUKAREAL
"
for item in $FILES; do
  id=${item%%:*}; y=${item#*:}; out="data/DAT_ASCII_XAUUSD_M1_${y}.csv"
  [ -s "$out" ] && { echo "skip $out"; continue; }
  curl -sSL --retry 3 --max-time 300 -o "$out" \
    "https://drive.usercontent.google.com/download?id=${id}&export=download&confirm=t" &
done
wait
for f in data/DAT_ASCII_XAUUSD_M1_*.csv; do
  head -c 1 "$f" | grep -q '[0-9]' || { echo "下载失败（非 CSV，可能是权限页）: $f"; rm -f "$f"; exit 1; }
done
ls -la data/*.csv | wc -l | xargs echo "文件数:"
