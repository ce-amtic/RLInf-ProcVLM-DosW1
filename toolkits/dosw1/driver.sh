#!/usr/bin/env bash
set -euo pipefail
name=rlinf-dosw1-right
case "${1:-status}" in
  start)
    test -e /sys/class/net/can_right
    if [[ -e /run/ruc-teleop/follow-desired ]]; then
      echo 'Disable the existing teleop follow stack before starting RLInf.' >&2
      exit 1
    fi
    if docker container inspect "$name" >/dev/null 2>&1; then
      docker start "$name"
    else
      if ss -H -lnt 'sport = :50053' | rg -q .; then
        echo 'Port 50053 already occupied; inspect the existing driver.' >&2
        exit 1
      fi
      docker run -d --name "$name" --network=host \
        registry.cn-shanghai.aliyuncs.com/discover-robotics/airbot-runtime:5.1.6 \
        ros2 run fsm fsm_node -i can_right -p 50053
    fi
    ready=false
    for ((attempt=0; attempt<120; attempt++)); do
      if ss -H -lnt 'sport = :50053' | rg -q .; then
        ready=true
        break
      fi
      if [[ "$(docker inspect --format '{{.State.Running}}' "$name")" != true ]]; then
        docker logs --tail 40 "$name" >&2
        exit 1
      fi
      sleep 0.25
    done
    if [[ "$ready" != true ]]; then
      echo 'Right driver did not open port 50053 within 30 seconds.' >&2
      docker logs --tail 40 "$name" >&2
      exit 1
    fi
    echo 'Right driver listening on 50053; run preflight.py --hardware to verify feedback.'
    ;;
  status) docker inspect --format '{{.State.Status}}' "$name"; ss -lnt 'sport = :50053' ;;
  stop) docker stop -t 10 "$name" ;;
  logs) docker logs --tail 80 "$name" ;;
  *) echo 'Usage: driver.sh start|status|stop|logs' >&2; exit 2 ;;
esac
