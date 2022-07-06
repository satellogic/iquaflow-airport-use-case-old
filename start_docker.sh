docker build -t airport .
docker run --rm --name air --gpus all -p 9197:9197 -v $(pwd):/airport/faster -v /Data/share:/Data/share -it airport