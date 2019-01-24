#!/bin/sh
# This script will create a portforward usin ssh
# Give remote IP with a remote port, this script whill bind it
# to the localport. Setup SSH on the server.

# Example:
# ---------------
# REMOTEIP=192.168.0.2
# REMOTEPORT=88
# LOCALPORT=8888
#
# ./tunnel-ip

REMOTEIP=$1
REMOTEPORT=$2
LOCALPORT=$3
USER=$4
PUBLICIP=$5

echo "Forwarding $REMOTEIP to port $LOCALPORT"
ssh -fNL $LOCALPORT:$REMOTEIP:$REMOTEPORT $USER@$PUBLICIP
echo -e "IP forwared \t http://localhost:$LOCALPORT"
