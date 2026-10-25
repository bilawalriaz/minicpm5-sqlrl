#!/bin/zsh
# Usage: watchdog.sh <pid> <logfile>. Kills <pid> if swap grows >300MB or its footprint passes 32GB.
pid=$1; log=$2
swap0=$(sysctl -n vm.swapusage | awk '{print $6}' | tr -d M)
while kill -0 $pid 2>/dev/null; do
  s=$(sysctl -n vm.swapusage | awk '{print $6}' | tr -d M)
  fp=$(footprint -p $pid 2>/dev/null | awk '/phys_footprint:/{v=$2; u=$3; if(u=="GB") v*=1024; if(u=="KB") v/=1024; print int(v); exit}')
  echo "$(date +%T) swap ${s}M footprint ${fp}MB" >> $log
  if [ "$(echo "$s - $swap0 > 300" | bc)" = 1 ] || [ "${fp:-0}" -gt 32768 ]; then echo "LIMIT HIT: swap $s footprint $fp" >> $log; kill $pid; fi
  sleep 4
done
