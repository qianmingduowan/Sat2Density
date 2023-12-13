CHECKPOINTS="cvusa/run-20230303_142752-2cqv8uh4.zip cvact/run-20230219_141512-2u87bj8w.zip"

if [ ! -d "wandb" ]; then
  mkdir wandb
fi

for checkpoint in $CHECKPOINTS ; do
    echo "Downloading $checkpoint";
    if [ ! -f "wandb/$checkpoint" ]; then
        wget https://github.com/sat2density/checkpoints/releases/download/$checkpoint -P wandb
    fi
    echo "Unzipping $checkpoint";
    if [ ! -d "wandb/${checkpoint%.*}" ]; then
        unzip wandb/$checkpoint -d wandb
    fi
done
