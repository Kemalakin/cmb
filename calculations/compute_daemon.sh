until ./compute.py; do
    echo "Server 'compute.py' crashed with exit code $?. Respawning.." >&2
    sleep 1
done