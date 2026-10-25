#!/bin/zsh
# Usage: watchdog.sh <pid> <logfile> [footprint-limit-MB] [swap-growth-limit-MB]
# Sends SIGTERM (the trainer then saves an interrupted checkpoint) after three consecutive
# checks over a limit: process footprint, swap growth since start, or critical memory pressure.
# Sends SIGKILL if the process is still alive 180 s later.
pid=$1; log=$2; fp_limit=${3:-30720}; swap_limit=${4:-3072}
swap0=$(sysctl -n vm.swapusage | awk '{print $6}' | tr -d M)
strikes=0
echo "$(date '+%F %T') watching $pid: footprint ${fp_limit}MB, swap growth ${swap_limit}MB, critical pressure" >> $log
while kill -0 $pid 2>/dev/null; do
  s=$(sysctl -n vm.swapusage | awk '{print $6}' | tr -d M)
  level=$(sysctl -n kern.memorystatus_vm_pressure_level)
  fp=$(footprint -p $pid 2>/dev/null | awk '/phys_footprint:/{v=$2; u=$3; if(u=="GB") v*=1024; if(u=="KB") v/=1024; print int(v); exit}')
  echo "$(date '+%F %T') swap ${s}M footprint ${fp}MB pressure ${level}" >> $log
  if [ "$(echo "$s - $swap0 > $swap_limit" | bc)" = 1 ] || [ "${fp:-0}" -gt $fp_limit ] || [ "$level" -ge 4 ]; then
    strikes=$((strikes + 1))
  else
    strikes=0
  fi
  if [ $strikes -ge 3 ]; then
    echo "$(date '+%F %T') LIMIT HIT: swap $s (start $swap0) footprint $fp pressure $level; SIGTERM" >> $log
    kill -TERM $pid
    for i in $(seq 36); do kill -0 $pid 2>/dev/null || exit 0; sleep 5; done
    echo "$(date '+%F %T') still alive; SIGKILL" >> $log
    kill -KILL $pid
    exit 0
  fi
  sleep 10
done
echo "$(date '+%F %T') process $pid exited" >> $log
