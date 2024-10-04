
#!/usr/bin/bash

kill $(cat scipyopt.pid)
kill $(cat run_task.pid)
while read pid; do
  kill $pid
done < geom.pids
