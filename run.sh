echo "UID=$(id -u)" > .env
echo "GID=$(id -g)" >> .env
echo "USER=$(whoami)" >> .env
echo "COMMANDS_PATH=$1" >> .env
docker-compose up