# cnam-ia-projet-smart

## Commands

```bash
# Create a new virtual environment -> already done
python3.11 -m venv .venv
```

```bash
# Activate the virtual environment
source .venv/bin/activate
```

```bash
# Install dependency
pip install -r requirements.txt
```

## Usage
Create the .secrets.toml file in the root directory based on .secrets-example.toml and fill in the values.

## Arguments
### Choose the pipeline to run
`--train` to train the model
`--infer` to use the model

### inferring options
`webcam` to use the webcam
`image` to use an image (require --path)
`video` to use a video (require --path)

### Examples
```bash
python main.py --train
python main.py --infer webcam
python main.py --infer video --path path/to/video.mp4
```

