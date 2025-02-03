

# Installation

```sh
uv add rouge4llm
```

Install [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) if you want to quantize the model by running the following command.

```sh
uv add bitsandbytes
```

# Example

```sh
python -m rouge4llm.commands.run \
  --model-name "meta-llama/Llama-3.2-1B-Instruct" \
  --sys-inst "Generate a one-senetnce summary of the given research paper with a focus on its research challenge in less than 25 words." \
  --dataset scitldr \
  --split test \
  --output-dir ./
```
