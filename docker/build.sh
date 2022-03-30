#!/bin/bash

echo 'Building Dockerfile with image name rl_games'
docker build --network=host -t sad-legacy .
