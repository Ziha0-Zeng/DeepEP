"""Opt-in Combine diagnostics, reusing the C++ benchmark's device record.

The last call is retained. Timestamps share an origin only within one rank;
different NPUs/hosts are NOT assumed to have synchronized device clocks.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import threading

_state = threading.local()
STAGES = (
    ('init', 'Entry', 'InitEnd'),
    ('stage_local', 'StageBegin', 'StageEnd'),
    ('notify', 'NotifyBegin', 'NotifyEnd'),
    ('wait_server_ready', 'ReadyBegin', 'ReadyEnd'),
    ('partial_M4', 'PartialBegin', 'PartialEnd'),
    ('final_M5', 'FinalBegin', 'FinalEnd'),
    ('drain_output', 'FinalEnd', 'DrainEnd'),
    ('direct_return', 'DirectBegin', 'DirectEnd'),
    ('tail_flush', 'PipeDone', 'FlushDone'),
)


def decode(raw, layout, rank, world, tokens, level, recv_cores, frequency_hz):
    if not math.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError('Counter frequency must be positive and finite')
    if not raw or not 0 < recv_cores <= len(raw):
        raise ValueError('Invalid Combine profile core count')
    idx = layout['indices']
    rows, events = [], []
    scale = 1e6 / frequency_hz
    origin = min(row[idx['Entry']] for row in raw)
    for core, row in enumerate(raw):
        if len(row) != layout['words']:
            raise ValueError('Invalid Combine profile record size')
        expected = dict(Schema=layout['schema'], Complete=1, Rank=rank, Core=core,
                        World=world, Tokens=tokens, Level=level)
        if any(row[idx[key]] != value for key, value in expected.items()):
            raise ValueError('Incomplete or mismatched Combine profile record')
        valid = row[idx['Valid']]
        if any(not valid & (1 << idx[key]) for key in ('Entry', 'FlushDone')):
            raise ValueError('Missing Combine profile boundaries')
        timestamps = {key: (row[index] - origin) * scale for key, index in idx.items()
                      if index < idx['StageRows'] and valid & (1 << index)}
        counters = {key: row[index] for key, index in idx.items() if index >= idx['StageRows']}
        role = 'send' if core < len(raw) - recv_cores else 'receive'
        intervals = {}
        for name, begin, end in STAGES:
            if begin not in timestamps or end not in timestamps:
                continue
            duration = timestamps[end] - timestamps[begin]
            if duration < 0:
                raise ValueError('Non-monotonic Combine profile clock')
            intervals[name] = duration
            events.append(dict(name=name, cat=role, ph='X', pid=rank, tid=core,
                               ts=timestamps[begin], dur=duration,
                               args=dict(rank_local_clock=True)))
        rows.append(dict(core=core, role=role, timestamps_us=timestamps,
                         intervals_us=intervals, counters=counters))
    return dict(schema=layout['schema'], rank=rank, tokens=tokens, level=level,
                frequency_hz=frequency_hz, cores=rows, raw=raw,
                note='Diagnostic overhead included. Each rank has an independent clock origin; do not infer cross-rank launch skew.',
                traceEvents=events)


class CombineTrace:
    def __init__(self, buffer, level=1, frequency_hz=1e9):
        from .._native import require_native
        import torch
        native = require_native('Combine profile')
        if not getattr(native, 'low_latency_combine_profile_enabled', False):
            raise RuntimeError('Rebuild and install both hosts with DEEPEP_LL_COMBINE_PROFILE=ON')
        if level not in (1, 2):
            raise ValueError('Combine profile level must be 1 or 2')
        self.layout = native.low_latency_combine_profile_layout()
        if self.layout['schema'] != 3:
            raise RuntimeError('Unsupported Combine profile schema')
        config = native.low_latency_core_config()
        cores = config['active_aiv']
        if not config['runtime_device_queried']:
            raise RuntimeError('Bind an NPU before creating CombineTrace')
        self.recv = native.low_latency_combine_recv_cores or cores // 2
        words = cores * self.layout['words']
        backing = torch.zeros(words + 63, dtype=torch.int64, device=f"npu:{config['device']}")
        offset = (-backing.data_ptr()) % 512 // 8
        self.tensor = backing.narrow(0, offset, words).view(cores, self.layout['words'])
        self.buffer, self.level, self.frequency_hz = buffer, level, frequency_hz
        self.launches = 0

    def save(self, directory, rank, world, tokens):
        if self.launches < 1:
            raise RuntimeError('No Combine call was captured')
        report = decode(self.tensor.cpu().tolist(), self.layout, rank, world, tokens,
                        self.level, self.recv, self.frequency_hz)
        report['launches_overwritten'] = self.launches - 1
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, value in (('profile.json', report),
                            ('timeline.json', dict(traceEvents=report['traceEvents'],
                                                   displayTimeUnit='us', note=report['note']))):
            with (directory / name).open('x', encoding='utf-8') as stream:
                json.dump(value, stream, indent=2)


@contextmanager
def capture(trace):
    active = getattr(_state, 'active', {})
    if id(trace.buffer) in active:
        raise RuntimeError('Nested Combine profiles are unsupported')
    _state.active = active
    active[id(trace.buffer)] = trace
    try:
        yield trace
    finally:
        active.pop(id(trace.buffer), None)


def attach(buffer, tensors):
    trace = getattr(_state, 'active', {}).get(id(buffer))
    if trace is None:
        return
    if trace.tensor.device != tensors['input'].device:
        raise ValueError('Combine profile must be on the input NPU')
    trace.launches += 1
    tensors['ll_combine_profile'] = trace.tensor
    tensors['ll_combine_profile_level'] = trace.level
