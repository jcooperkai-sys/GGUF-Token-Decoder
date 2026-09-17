# GGUF Token Decoder

Turn a local language model's raw token IDs back into real text, using
**only the model file itself** -- no `transformers`, no downloading a
separate tokenizer, no network access at all.

## Why

Every local model that runs as a `.gguf` file (Ollama, LM Studio,
llama.cpp, koboldcpp, text-generation-webui, ...) carries its own complete
tokenizer vocabulary embedded inside that same file, as plain metadata.
Whatever the model actually emits -- its final answer, or a reasoning
model's separate "thinking" stream -- is just a sequence of integer token
IDs, each one an index into that vocabulary. This tool reads that
vocabulary directly out of the `.gguf` file and lets you decode any token
ID (or dump the whole table) back to the real text it represents.

```
25125  ->  '_A'
51902  ->  'Ġbanana'
```

(The `Ġ` you'll see on many tokens is a real, standard marker most GPT-2/BPE-style
tokenizers use for "this token starts a new word" -- it isn't a bug or
mangled text, it's how the vocabulary really stores that information.
Pass `--clean` to `decode` to reverse it back to a normal space, the way
`decode_text_clean()` does below.)

## Install

No dependencies. Python 3.10+, standard library only.

```bash
git clone https://github.com/<you>/gguf-token-decoder.git
cd gguf-token-decoder
python3 gguf_tokens.py --help
```

## Usage

```bash
# Decode specific token IDs
python3 gguf_tokens.py ./your-model.gguf decode 25125 51902 100

# Reconstruct real text from a sequence of IDs (raw tokenizer pieces)
python3 gguf_tokens.py ./your-model.gguf decode 51902 67403 --join

# Same, but reversed back to normal human-readable text -- real spaces
# and newlines instead of the tokenizer's own Ġ/Ċ/▁ markers
python3 gguf_tokens.py ./your-model.gguf decode 51902 67403 --clean

# Search the real vocabulary for every token containing a substring
python3 gguf_tokens.py ./your-model.gguf find banana

# Dump the entire real id -> token table as JSON
python3 gguf_tokens.py ./your-model.gguf dump --out vocab.json

# Print the tokenizer metadata the file actually carries
python3 gguf_tokens.py ./your-model.gguf info
```

Finding a real `.gguf` file to point this at:

- **Ollama**: `~/.ollama/models/manifests/...` has a JSON manifest per
  model naming a `sha256-...` digest under `application/vnd.ollama.image.model`
  -- that digest is the real filename under `~/.ollama/models/blobs/`.
- **LM Studio / llama.cpp / koboldcpp**: you already have a plain `.gguf`
  file on disk; point this straight at it.

## As a library

```python
from gguf_tokens import load_vocab

vocab = load_vocab("your-model.gguf")
print(len(vocab), "real tokens")
print(vocab.decode_id(51902))          # 'Ġbanana'
print(vocab.decode_text([51902, 67403]))       # 'Ġbananaġbananas' (raw tokenizer pieces)
print(vocab.decode_text_clean([51902, 67403])) # ' banana bananas' (real, readable text)
for tid, tok in vocab.find("anana"):
    print(tid, tok)
```

## How it works

A `.gguf` file starts with a metadata section (key/value pairs -- model
name, context length, quantization info, and the tokenizer's own vocab
array under the key `tokenizer.ggml.tokens`) followed by the actual
tensor weight data, which can be many gigabytes. This tool parses only
the metadata header and stops -- it never reads the tensor data, so it
stays fast and light even against a large model file. See the [GGUF
format spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
for the full layout.

## License

MIT -- see [LICENSE](LICENSE).
