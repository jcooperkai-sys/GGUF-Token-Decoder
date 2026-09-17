"""gguf_tokens -- read a GGUF model file's own embedded tokenizer vocabulary
and turn raw token IDs back into real text, with zero ML framework
dependencies (pure Python, stdlib only).

A GGUF file (the format llama.cpp / Ollama / LM Studio all use for local
models) carries its full tokenizer vocabulary as plain metadata inside the
file itself -- every token ID a model ever emits (in its normal reply, or
in a "thinking"/chain-of-thought stream) is just an index into that same
vocabulary. This module reads that metadata directly, without loading the
multi-gigabyte weight tensors, and without needing `transformers` or the
original model's tokenizer.json.

GGUF format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""
from __future__ import annotations

import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

GGUF_MAGIC = b"GGUF"

# gguf_metadata_value_type enum (see the format spec linked above)
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16, _T_UINT32, _T_INT32, _T_FLOAT32, \
    _T_BOOL, _T_STRING, _T_ARRAY, _T_UINT64, _T_INT64, _T_FLOAT64 = range(13)

_SCALAR_FORMATS = {
    _T_UINT8: ("B", 1), _T_INT8: ("b", 1),
    _T_UINT16: ("H", 2), _T_INT16: ("h", 2),
    _T_UINT32: ("I", 4), _T_INT32: ("i", 4),
    _T_FLOAT32: ("f", 4),
    _T_UINT64: ("Q", 8), _T_INT64: ("q", 8),
    _T_FLOAT64: ("d", 8),
}


def _gpt2_byte_decoder() -> dict[str, int]:
    """The real, standard GPT-2 byte-level BPE mapping (from OpenAI's own
    reference implementation), inverted. A byte-level BPE tokenizer never
    operates on raw bytes directly -- it first remaps every one of the 256
    possible byte values to a printable unicode character (so whitespace,
    control characters, etc. all become visible characters the BPE merge
    algorithm can work with), including the real, correct ones you saw:
    0x20 (space) -> 'Ġ', 0x0A (newline) -> 'Ċ'. This is that same mapping,
    inverted, so a decoded token string can be turned back into the real
    original bytes -- not a guess at "probably a space," the actual
    reverse of the actual forward mapping every GPT-2-style tokenizer
    (GPT-2, GPT-J, StarCoder, Qwen, DeepSeek, and others) really uses."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_encoder = {b: chr(c) for b, c in zip(bs, cs)}
    return {ch: b for b, ch in byte_encoder.items()}


_GPT2_BYTE_DECODER = _gpt2_byte_decoder()


class GGUFParseError(Exception):
    """The file isn't a real, well-formed GGUF file at the point this failed."""


@dataclass
class GGUFVocab:
    """The decoded result: id -> token string, plus whatever tokenizer
    metadata the file carried (useful context, not required to decode)."""
    tokens: list[str]
    tokenizer_model: str | None = None          # e.g. "gpt2", "llama", "bpe"
    token_types: list[int] = field(default_factory=list)
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    all_metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.tokens)

    def decode_id(self, token_id: int) -> str:
        """The real token text for one real token ID -- raises IndexError
        for an ID outside this file's own real vocabulary, rather than
        silently returning a placeholder for something that was never
        actually in it."""
        if token_id < 0 or token_id >= len(self.tokens):
            raise IndexError(
                f"token id {token_id} is outside this model's real vocab "
                f"(size {len(self.tokens)})"
            )
        return self.tokens[token_id]

    def decode_ids(self, token_ids: list[int]) -> list[str]:
        return [self.decode_id(t) for t in token_ids]

    def decode_text(self, token_ids: list[int]) -> str:
        """Concatenates each real token's piece in order. Many tokenizers
        (BPE/SentencePiece) use a real marker byte (commonly U+2581 '▁',
        sometimes a literal leading space) for a real word-start boundary
        inside each token piece -- this is left exactly as the vocab
        stored it rather than guessed at, so the output is the real,
        exact reconstruction, not a cleaned-up approximation."""
        return "".join(self.decode_ids(token_ids))

    def decode_text_clean(self, token_ids: list[int]) -> str:
        """The real, human-readable text -- not the raw tokenizer piece
        markers. Two real tokenizer families need two different real
        reversals, and this picks the right one from the file's own
        tokenizer.ggml.model metadata rather than guessing:

        - Byte-level BPE (GPT-2 and everything built the same way --
          GPT-J, StarCoder, Qwen, DeepSeek, ...): every REAL (non-special)
          token piece is a sequence of the real remapped-byte characters
          (see _gpt2_byte_decoder). A real special/control token (e.g.
          <think>, <｜User｜>) is never byte-remapped -- it's already
          literal text -- so each piece is converted to real raw bytes
          individually (falling back to a direct UTF-8 encode for a
          special token), and only the FULL resulting byte sequence is
          UTF-8-decoded, once, at the very end. Decoding piece-by-piece
          instead would break on a real multi-byte character (e.g. an
          emoji, or non-Latin text) that a real BPE tokenizer legitimately
          split across two adjacent token pieces.
        - SentencePiece (LLaMA, Mistral, Gemma, ...): a word boundary is a
          real literal '▁' (U+2581) character, not a byte remapping --
          this just turns each one into a real space.

        Falls back to the raw decode_text if the file's tokenizer type
        isn't one of these two (rather than silently mangling text from a
        family this hasn't been taught the real reversal for)."""
        model = (self.tokenizer_model or "").lower()
        pieces = self.decode_ids(token_ids)
        if model in ("gpt2", "gpt-2", "bpe"):
            raw_bytes = bytearray()
            for piece in pieces:
                try:
                    raw_bytes += bytes(_GPT2_BYTE_DECODER[ch] for ch in piece)
                except KeyError:
                    raw_bytes += piece.encode("utf-8")
            return bytes(raw_bytes).decode("utf-8", errors="replace")
        if model in ("llama", "sentencepiece", "spm"):
            return "".join(pieces).replace("▁", " ")
        return "".join(pieces)

    def find(self, substring: str, limit: int = 50) -> list[tuple[int, str]]:
        """Every real (id, token) pair whose token text actually contains
        substring -- e.g. find("Banana") to see every real vocab entry
        that could ever produce that word, including sub-word pieces."""
        hits = []
        for tid, tok in enumerate(self.tokens):
            if substring in tok:
                hits.append((tid, tok))
                if len(hits) >= limit:
                    break
        return hits


def _read_string(f) -> str:
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length).decode("utf-8", errors="replace")


def _read_scalar(f, value_type: int):
    if value_type == _T_BOOL:
        return struct.unpack("<B", f.read(1))[0] != 0
    if value_type == _T_STRING:
        return _read_string(f)
    if value_type not in _SCALAR_FORMATS:
        raise GGUFParseError(f"unknown GGUF metadata value type {value_type}")
    fmt, size = _SCALAR_FORMATS[value_type]
    return struct.unpack("<" + fmt, f.read(size))[0]


def _read_value(f, value_type: int):
    if value_type == _T_ARRAY:
        (elem_type,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        return [_read_value(f, elem_type) for _ in range(count)]
    return _read_scalar(f, value_type)


def load_vocab(path: str | Path) -> GGUFVocab:
    """Reads ONLY the metadata section of a real GGUF file -- never touches
    the (often multi-GB) tensor data that follows it, so this is fast and
    memory-light even against a real full-size model file."""
    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise GGUFParseError(
                f"{path} does not start with the real GGUF magic bytes "
                f"(got {magic!r}) -- this isn't a real GGUF file"
            )
        (version,) = struct.unpack("<I", f.read(4))
        if version < 2:
            raise GGUFParseError(f"GGUF version {version} predates the metadata layout this reads")
        (_tensor_count,) = struct.unpack("<Q", f.read(8))
        (kv_count,) = struct.unpack("<Q", f.read(8))

        metadata: dict = {}
        for _ in range(kv_count):
            key = _read_string(f)
            (value_type,) = struct.unpack("<I", f.read(4))
            metadata[key] = _read_value(f, value_type)

    tokens = metadata.get("tokenizer.ggml.tokens")
    if not tokens:
        raise GGUFParseError(
            f"{path} has real GGUF metadata but no tokenizer.ggml.tokens "
            f"array -- it may be a non-text (e.g. vision-encoder-only) file"
        )
    return GGUFVocab(
        tokens=tokens,
        tokenizer_model=metadata.get("tokenizer.ggml.model"),
        token_types=metadata.get("tokenizer.ggml.token_type", []),
        bos_token_id=metadata.get("tokenizer.ggml.bos_token_id"),
        eos_token_id=metadata.get("tokenizer.ggml.eos_token_id"),
        all_metadata=metadata,
    )


def _cli(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="gguf-tokens",
        description="Decode raw token IDs back to real text using a GGUF model's own embedded vocabulary.",
    )
    parser.add_argument("gguf_file", help="Path to a real local .gguf model file")
    sub = parser.add_subparsers(dest="command", required=True)

    p_decode = sub.add_parser("decode", help="Decode one or more real token IDs to their real text")
    p_decode.add_argument("ids", nargs="+", type=int, help="Real token IDs, e.g. 25125 881 204")
    p_decode.add_argument("--join", action="store_true", help="Concatenate the decoded pieces into one real string instead of listing each pair")
    p_decode.add_argument("--clean", action="store_true", help="Reverse the tokenizer's real byte/space markers (Ġ/Ċ/▁) back to normal readable text. Implies --join.")

    p_find = sub.add_parser("find", help="Search the real vocab for every token containing a substring")
    p_find.add_argument("substring")
    p_find.add_argument("--limit", type=int, default=50)

    p_dump = sub.add_parser("dump", help="Write the real, full id->token table as JSON")
    p_dump.add_argument("--out", default="-", help="Output path, or - for stdout")

    p_info = sub.add_parser("info", help="Print the real tokenizer metadata this file actually carries")

    args = parser.parse_args(argv)

    try:
        vocab = load_vocab(args.gguf_file)
    except (GGUFParseError, FileNotFoundError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    if args.command == "decode":
        try:
            if args.clean:
                print(vocab.decode_text_clean(args.ids))
                return 0
            pieces = vocab.decode_ids(args.ids)
        except IndexError as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        if args.join:
            print("".join(pieces))
        else:
            for tid, piece in zip(args.ids, pieces):
                print(f"{tid}\t{piece!r}")

    elif args.command == "find":
        hits = vocab.find(args.substring, args.limit)
        if not hits:
            print(f"no real vocab entries contain {args.substring!r}", file=sys.stderr)
            return 1
        for tid, tok in hits:
            print(f"{tid}\t{tok!r}")

    elif args.command == "dump":
        table = {i: t for i, t in enumerate(vocab.tokens)}
        text = json.dumps(table, ensure_ascii=False, indent=2)
        if args.out == "-":
            print(text)
        else:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"wrote {len(table)} real vocab entries to {args.out}", file=sys.stderr)

    elif args.command == "info":
        print(f"real vocab size:      {len(vocab)}")
        print(f"tokenizer model:      {vocab.tokenizer_model}")
        print(f"bos_token_id:         {vocab.bos_token_id}")
        print(f"eos_token_id:         {vocab.eos_token_id}")
        print(f"other metadata keys:  {len(vocab.all_metadata)}")

    return 0


def _cli_entry() -> None:
    """Console-script entry point (see pyproject.toml) -- reads real argv."""
    raise SystemExit(_cli(sys.argv[1:]))


if __name__ == "__main__":
    _cli_entry()
