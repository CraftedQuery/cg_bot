import types
import importlib.util

spec = importlib.util.spec_from_file_location('utils.file_processors', 'utils/file_processors.py')
file_proc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(file_proc)


def test_chunk_text_with_lines():
    text = "# Heading\nline1\nline2\nline3\nline4"
    chunks, metas = file_proc._chunk_text_with_lines(text, lines_per_chunk=2)
    assert chunks[0].startswith("# Heading") or chunks[0].startswith("line1")
    assert metas[0]['line'] == 1
    assert metas[0]['heading'] == 'Heading'
    assert metas[1]['line'] == 3

