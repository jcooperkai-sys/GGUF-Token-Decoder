"""Real tests against a real, hand-built minimal GGUF byte stream -- not a
downloaded model file (so these run instantly, anywhere, with no fixture
files to fetch), but genuinely exercising the real binary format, not a
mocked-out parser."""
import struct
import tempfile
from pathlib import Path

import pytest

from gguf_tokens import GGUFParseError, load_vocab

_T_UINT32, _T_STRING, _T_ARRAY, _T_UINT64 = 4, 8, 9, 10


def _write_string(buf: bytearray, s: str) -> None:
    encoded = s.encode("utf-8")
    buf += struct.pack("<Q", len(encoded))
    buf += encoded


def _write_kv_string(buf: bytearray, key: str, value: str) -> None:
    _write_string(buf, key)
    buf += struct.pack("<I", _T_STRING)
    _write_string(buf, value)


def _write_kv_uint32(buf: bytearray, key: str, value: int) -> None:
    _write_string(buf, key)
    buf += struct.pack("<I", _T_UINT32)
    buf += struct.pack("<I", value)


def _write_kv_string_array(buf: bytearray, key: str, values: list[str]) -> None:
    _write_string(buf, key)
    buf += struct.pack("<I", _T_ARRAY)
    buf += struct.pack("<I", _T_STRING)
    buf += struct.pack("<Q", len(values))
    for v in values:
        _write_string(buf, v)


def _build_minimal_gguf(tokens: list[str]) -> bytes:
    kvs = bytearray()
    _write_kv_string(kvs, "tokenizer.ggml.model", "gpt2")
    _write_kv_string_array(kvs, "tokenizer.ggml.tokens", tokens)
    _write_kv_uint32(kvs, "tokenizer.ggml.bos_token_id", 0)
    _write_kv_uint32(kvs, "tokenizer.ggml.eos_token_id", 1)

    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 4)
    return bytes(header) + bytes(kvs)


@pytest.fixture()
def real_gguf_path(tmp_path: Path) -> Path:
    tokens = ["<bos>", "<eos>", "Hello", "Ġworld", "Ġbanana", "Ġbananas", "!"]
    data = _build_minimal_gguf(tokens)
    path = tmp_path / "tiny-real.gguf"
    path.write_bytes(data)
    return path


def test_load_vocab_reads_real_tokens(real_gguf_path):
    vocab = load_vocab(real_gguf_path)
    assert len(vocab) == 7
    assert vocab.tokenizer_model == "gpt2"
    assert vocab.bos_token_id == 0
    assert vocab.eos_token_id == 1


def test_decode_id_returns_real_token(real_gguf_path):
    vocab = load_vocab(real_gguf_path)
    assert vocab.decode_id(4) == "Ġbanana"
    assert vocab.decode_id(2) == "Hello"


def test_decode_id_out_of_range_raises(real_gguf_path):
    vocab = load_vocab(real_gguf_path)
    with pytest.raises(IndexError):
        vocab.decode_id(999)


def test_decode_text_concatenates_real_pieces(real_gguf_path):
    vocab = load_vocab(real_gguf_path)
    assert vocab.decode_text([2, 3]) == "HelloĠworld"


def test_find_matches_real_substring(real_gguf_path):
    vocab = load_vocab(real_gguf_path)
    hits = vocab.find("anana")
    assert (4, "Ġbanana") in hits
    assert (5, "Ġbananas") in hits
    assert len(hits) == 2


def test_find_respects_limit(real_gguf_path):
    vocab = load_vocab(real_gguf_path)
    hits = vocab.find("Ġ", limit=2)
    assert len(hits) == 2


def test_not_a_real_gguf_file_raises(tmp_path):
    bogus = tmp_path / "not-a-model.bin"
    bogus.write_bytes(b"NOTGGUF" + b"\x00" * 32)
    with pytest.raises(GGUFParseError):
        load_vocab(bogus)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_vocab(tmp_path / "does-not-exist.gguf")
