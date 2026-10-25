"""Authorized, single-pass fixed NIQE/MUSIQ scoring of the audited 742 PNGs.

Preparation never calls this file. The coordinator must provide a new, explicit
authorization binding this exact script, protocol, environment and preflight.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
PREP = Path(__file__).resolve().parent
ROOT = PREP.parents[2]
os.environ['TORCH_HOME'] = str(PREP / 'torch_cache')
os.environ['HF_HOME'] = str(PREP / 'hf_cache')

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata as meta
import io
import json
import math
import random
import statistics
import sys
import time
import traceback
import numpy as np
from PIL import Image
import pyiqa
import torch

GROUPS = [('Input', ''), ('A', '20261004'), ('SUM4', '20261004'), ('B', '20261004'),
          ('MERGED_LUM_PCGRAD', '20261004'), ('MERGED_LUM_PCGRAD', '20261005'),
          ('MERGED_LUM_PCGRAD', '20261006')]
METRICS = ['niqe', 'musiq']
ABSOLUTE_DEADLINE = 1792296000  # 2026-10-18 12:00:00 Asia/Shanghai


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def checked_bytes(row):
    p = (ROOT / row['path']).resolve()
    assert p.is_relative_to(ROOT.resolve()), row['path']
    data = p.read_bytes()
    assert len(data) == int(row['bytes']), row['sample_id']
    assert hashlib.sha256(data).hexdigest().upper() == row['sha256'].upper(), row['sample_id']
    return data


def freeze_runtime():
    assert sys.prefix != sys.base_prefix and Path(sys.prefix).resolve() == (PREP / 'venv').resolve()
    assert meta.version('pyiqa') == '0.1.16'
    assert torch.__version__ == '2.8.0+cu128'
    assert torch.cuda.is_available() and '5060' in torch.cuda.get_device_name(0)
    random.seed(20261016)
    np.random.seed(20261016)
    torch.manual_seed(20261016)
    torch.cuda.manual_seed_all(20261016)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def validate_authorization(auth):
    assert auth['action'] == 'SCORE_EXISTING_REAL_NIGHT_742_ONCE'
    assert auth['metrics'] == METRICS and auth['records'] == 742
    assert auth['allow_real_scoring'] is True
    assert auth['device'] == 'cuda:0' and auth['seed'] == 20261016
    assert auth['deadline_epoch'] <= ABSOLUTE_DEADLINE
    assert time.time() < auth['deadline_epoch']
    expected = {
        'runner_sha256': Path(__file__),
        'protocol_sha256': PREP / 'IQA_PROTOCOL_ZH.md',
        'bindings_sha256': PREP / 'IQA_INPUT_BINDINGS.csv',
        'environment_sha256': PREP / 'ENVIRONMENT_PREPARATION.json',
        'preflight_sha256': PREP / 'SYNTHETIC_PREFLIGHT.json',
        'resource_lock_sha256': PREP / 'OFFICIAL_RESOURCE_LOCK.json',
    }
    for key, path in expected.items():
        assert auth[key].lower() == sha(path), key
    preflight = read_json(expected['preflight_sha256'])
    assert preflight['status'] == 'PASS'
    assert preflight['real_images_opened'] == preflight['real_scores_produced'] == 0
    assert preflight['protocol_sha256'] == auth['protocol_sha256'].lower()
    assert preflight['source_sha256'] == sha(PREP / 'synthetic_preflight.py')
    environment = read_json(expected['environment_sha256'])
    assert environment['status'] == 'RUNTIME_IMPORT_PASS_SYNTHETIC_PREFLIGHT_PENDING'
    assert environment['isolated_no_system_site_packages'] is True
    assert time.time() + preflight['projected_742_seconds_with_1_5x_and_300s_io'] < auth['deadline_epoch']
    lock = read_json(expected['resource_lock_sha256'])
    for item in lock['resources']:
        p = PREP / 'resources' / item['name']
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256']
    for relative, expected_sha in lock['pyiqa_python_sources_sha256'].items():
        assert sha(PREP / 'venv/Lib/site-packages' / relative) == expected_sha
    return preflight


def model_pair():
    niqe = pyiqa.create_metric('niqe', device='cuda:0', seed=20261016,
        test_y_channel=True, color_space='yiq', crop_border=0, version='original',
        pretrained_model_path=str(PREP / 'resources/niqe_modelparameters.mat'))
    musiq = pyiqa.create_metric('musiq', device='cuda:0', seed=20261016,
        pretrained='koniq10k', longer_side_lengths=[224, 384], max_seq_len_from_original_res=-1,
        pretrained_model_path=str(PREP / 'resources/musiq_koniq_ckpt-e95806b9.pth'))
    for model in (niqe, musiq):
        model.eval().requires_grad_(False)
        assert not model.net.training and not any(p.requires_grad for p in model.net.parameters())
    assert bool(niqe.lower_better) and not bool(musiq.lower_better)
    return {'niqe': niqe, 'musiq': musiq}


def append_event(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    handle.flush()
    os.fsync(handle.fileno())


def summarize(rows):
    baseline = {r['sample_id']: r for r in rows if r['condition'] == 'Input'}
    assert len(baseline) == 106
    summaries, sequence_rows, paired = [], [], []
    for condition, seed in GROUPS:
        selected = [r for r in rows if r['condition'] == condition and r['seed'] == seed]
        assert len(selected) == 106
        result = {'condition': condition, 'seed': seed, 'n_frames': 106}
        for metric in METRICS:
            values = [r[metric] for r in selected]
            deltas = [r[metric] - baseline[r['sample_id']][metric] for r in selected]
            result.update({metric + '_mean': statistics.mean(values),
                           metric + '_median': statistics.median(values),
                           metric + '_delta_vs_input_mean': statistics.mean(deltas),
                           metric + '_delta_vs_input_median': statistics.median(deltas)})
        summaries.append(result)
        for sequence in ('GOPR0351', 'GOPR0356'):
            subset = [r for r in selected if r['sequence'] == sequence]
            entry = {'condition': condition, 'seed': seed, 'sequence': sequence, 'n_frames': len(subset)}
            for metric in METRICS:
                entry[metric + '_mean'] = statistics.mean(r[metric] for r in subset)
                entry[metric + '_median'] = statistics.median(r[metric] for r in subset)
            sequence_rows.append(entry)
        for row in selected:
            entry = {k: row[k] for k in ('condition', 'seed', 'sample_id', 'sequence', 'sha256')}
            for metric in METRICS:
                entry[metric + '_delta_vs_input'] = row[metric] - baseline[row['sample_id']][metric]
            paired.append(entry)
    merged = [r for r in summaries if r['condition'] == 'MERGED_LUM_PCGRAD']
    assert len(merged) == 3
    across_seeds = {'condition': 'MERGED_LUM_PCGRAD', 'seeds': [20261004, 20261005, 20261006],
                    'n_training_seeds': 3, 'sd_definition': 'sample standard deviation, ddof=1'}
    for metric in METRICS:
        for field in ('mean', 'median', 'delta_vs_input_mean', 'delta_vs_input_median'):
            key = metric + '_' + field
            across_seeds[key + '_across_seeds_mean'] = statistics.mean(r[key] for r in merged)
            across_seeds[key + '_across_seeds_sample_sd'] = statistics.stdev(r[key] for r in merged)
    return {'conditions': summaries, 'sequences': sequence_rows, 'merged_three_seed': across_seeds,
            'unique_original_frames': 106, 'sequences_count': 2, 'independent_image_claim': False}, paired


def write_csv_new(path, rows):
    assert rows
    with Path(path).open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--authorization', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    auth = read_json(args.authorization)
    validate_authorization(auth)
    freeze_runtime()
    output = args.output.resolve()
    assert output.is_relative_to(PREP.parent.resolve()) and output != PREP.resolve()
    assert str(output) == str(Path(auth['output_directory']).resolve())
    assert not output.exists(), 'No repeats, overwrite, or resume permitted'
    # All identities are checked before either model is instantiated or any score.
    with (PREP / 'IQA_INPUT_BINDINGS.csv').open(encoding='utf-8-sig', newline='') as f:
        bindings = list(csv.DictReader(f))
    assert len(bindings) == 742
    assert [(r['condition'], r['seed']) for r in bindings[::106]] == GROUPS
    assert Counter((r['condition'], r['seed']) for r in bindings) == Counter({g: 106 for g in GROUPS})
    ids = [r['sample_id'] for r in bindings[:106]]
    for offset in range(0, 742, 106):
        assert [r['sample_id'] for r in bindings[offset:offset + 106]] == ids
    for row in bindings:
        checked_bytes(row)
    output.mkdir(exist_ok=False)
    write_json_new(output / 'RUN_AUTHORIZATION_COPY.json', auth)
    write_json_new(output / 'RUN_STARTED.json', {'time_utc': datetime.now(timezone.utc).isoformat(),
        'pid': os.getpid(), 'authorization_sha256': sha(args.authorization),
        'input_records_sha256': sha(PREP / 'IQA_INPUT_BINDINGS.csv'),
        'runner_sha256': sha(Path(__file__)), 'gpu': torch.cuda.get_device_name(0),
        'cuda_free_total_bytes': list(torch.cuda.mem_get_info()), 'deadline_epoch': auth['deadline_epoch']})
    complete = []
    status = {'status': 'STARTED', 'completed_records': 0, 'expected_records': 742, 'metric_events': 0}
    start = time.perf_counter()
    try:
        models = model_pair()
        torch.cuda.reset_peak_memory_stats()
        with (output / 'METRIC_EVENTS.jsonl').open('x', encoding='utf-8') as event_log, \
             (output / 'COMPLETE_RECORDS.jsonl').open('x', encoding='utf-8') as completed_log:
            for index, row in enumerate(bindings):
                assert time.time() < auth['deadline_epoch'], 'FAILED_INCOMPLETE: hard deadline'
                data = checked_bytes(row)
                with Image.open(io.BytesIO(data)) as image:
                    assert image.format == 'PNG' and image.mode == 'RGB' and image.size == (1920, 1080)
                    array = np.array(image, dtype=np.uint8, copy=True)
                inp = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous().to(dtype=torch.float32) / 255.0
                assert tuple(inp.shape) == (1, 3, 1080, 1920) and torch.isfinite(inp).all()
                assert 0 <= inp.min() <= inp.max() <= 1
                entry = dict(row)
                entry['index'] = index
                for metric_name in METRICS:
                    assert time.time() < auth['deadline_epoch'], 'FAILED_INCOMPLETE: hard deadline'
                    torch.cuda.synchronize()
                    call_start = time.perf_counter()
                    with torch.inference_mode():
                        value = models[metric_name](inp)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - call_start
                    assert value.numel() == 1 and torch.isfinite(value).all()
                    score = float(value.item())
                    assert math.isfinite(score)
                    entry[metric_name] = score
                    entry[metric_name + '_seconds'] = seconds
                    append_event(event_log, {'index': index, 'condition': row['condition'], 'seed': row['seed'],
                        'sample_id': row['sample_id'], 'sequence': row['sequence'], 'sha256': row['sha256'],
                        'metric': metric_name, 'value': score, 'seconds': seconds,
                        'time_utc': datetime.now(timezone.utc).isoformat()})
                    status['metric_events'] += 1
                entry['pyiqa_version'] = '0.1.16'
                entry['protocol_sha256'] = auth['protocol_sha256']
                entry['resource_lock_sha256'] = auth['resource_lock_sha256']
                complete.append(entry)
                append_event(completed_log, entry)
                status['completed_records'] = len(complete)
                if len(complete) % 20 == 0 or len(complete) % 106 == 0:
                    print(json.dumps({'completed': len(complete), 'expected': 742,
                                      'elapsed_seconds': time.perf_counter() - start}), flush=True)
        assert len(complete) == 742 and status['metric_events'] == 1484
        write_csv_new(output / 'PER_IMAGE.csv', complete)
        summary, paired = summarize(complete)
        write_json_new(output / 'SUMMARY.json', summary)
        write_csv_new(output / 'PAIRED_VS_INPUT.csv', paired)
        status['status'] = 'COMPLETE_742_1484'
        status['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        status['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    except Exception as error:
        status['status'] = 'FAILED_INCOMPLETE'
        status['error'] = repr(error)
        status['traceback'] = traceback.format_exc()
        status['failed_index'] = len(complete)
    finally:
        status['elapsed_seconds'] = time.perf_counter() - start
        status['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json_new(output / 'RUN_RESULT.json', status)
        manifest = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)}
                    for p in sorted(output.iterdir()) if p.is_file()]
        write_csv_new(output / 'OUTPUT_MANIFEST.csv', manifest)
        print(json.dumps(status, ensure_ascii=False), flush=True)
    if status['status'] != 'COMPLETE_742_1484':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
