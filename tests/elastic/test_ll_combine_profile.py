"""Host-only validation of the diagnostic schema and independent clock origins."""
from pathlib import Path
import re
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[2]
decode = runpy.run_path(str(ROOT / 'deep_ep/utils/ll_combine_profile.py'))['decode']
header = (ROOT / 'csrc/kernels/combine/low_latency/combine_profile_layout.h').read_text()
names = re.search(r'enum Word : uint32_t \{(.*?)\};', header, re.S).group(1)
indices = {name.strip(): i for i, name in enumerate(names.split(',')) if name.strip() != 'Count'}
LAYOUT = dict(schema=3, words=192, indices=indices)


def records(offset=10000):
    result = []
    for core in range(2):
        row = [0] * 192
        for name, value in dict(Entry=offset+core*100, InitEnd=offset+core*100+2000,
                                PipeDone=offset+50000, FlushDone=offset+51000,
                                Rank=4, World=8, Tokens=32, Core=core,
                                Complete=1, Schema=3, Level=1).items():
            row[indices[name]] = value
        row[indices['Valid']] = sum(1 << indices[key] for key in ('Entry', 'InitEnd', 'PipeDone', 'FlushDone'))
        result.append(row)
    return result


def parse(raw, **kwargs):
    return decode(raw, LAYOUT, 4, 8, 32, 1, 1, kwargs.get('frequency_hz', 1e9))


def test_rank_local_timeline_preserves_core_entry_skew():
    report = parse(records())
    assert report['cores'][0]['intervals_us']['init'] == 2
    assert report['cores'][1]['timestamps_us']['Entry'] == pytest.approx(0.1)
    assert report['cores'][1]['role'] == 'receive'
    assert report['traceEvents'] == parse(records(9000000000))['traceEvents']


@pytest.mark.parametrize('field,value', [('Complete', 0), ('Schema', 2), ('Rank', 0),
                                         ('Core', 1), ('World', 16), ('Tokens', 72), ('Level', 2), ('Valid', 0)])
def test_reject_incomplete_or_wrong_identity(field, value):
    raw = records()
    raw[0][indices[field]] = value
    with pytest.raises(ValueError):
        parse(raw)


@pytest.mark.parametrize('frequency', [0, -1, float('nan'), float('inf')])
def test_reject_invalid_frequency(frequency):
    with pytest.raises(ValueError):
        parse(records(), frequency_hz=frequency)


def test_reject_malformed_or_backward_clock():
    with pytest.raises(ValueError):
        parse([[0]])
    raw = records()
    raw[0][indices['InitEnd']] = 1
    with pytest.raises(ValueError):
        parse(raw)
