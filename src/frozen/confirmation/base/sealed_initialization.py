"""V2 zero-step state transport; keeps the original constructor and RNG consumption."""
import hashlib
import json
import random
import numpy as np
import torch
from portable import safe, sha, scratch_cpu, data_helpers
import core


class IdentityError(RuntimeError):
    def __init__(self, check, expected, actual):
        self.detail = {'check': check, 'expected': expected, 'actual': actual}
        super().__init__(json.dumps(self.detail, ensure_ascii=False))


def require(check, expected, actual):
    if expected != actual:
        raise IdentityError(check, expected, actual)


def raw_sha(tensor):
    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def rng_identity():
    # No CUDA RNG reads/initialization. The original worker seeds CUDA separately.
    ns = np.random.get_state()
    return {'python': hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
            'numpy': hashlib.sha256(ns[1].tobytes()+repr((ns[0],*ns[2:])).encode()).hexdigest(),
            'torch_cpu': raw_sha(torch.get_rng_state())}


def legacy_sha(model):
    return data_helpers()['tensor_sha'](core.flatten([p.detach() for p in model.parameters()]))


def inventory(model):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    state = model.state_dict()
    require('state_dict_covers_all_parameters_and_buffers', sorted(set(params)|set(buffers)), sorted(state))
    rows = []
    for name, value in state.items():
        rows.append({'name': name, 'role': 'parameter' if name in params else 'buffer',
                     'shape': list(value.shape), 'dtype': str(value.dtype), 'numel': value.numel(),
                     'bytes': value.numel()*value.element_size(), 'sha256': raw_sha(value)})
    return rows


def verify_state_mapping(state, identity, prefix):
    wanted = identity['tensors']
    require(prefix+'.ordered_state_keys', [r['name'] for r in wanted], list(state.keys()))
    for row in wanted:
        value = state[row['name']]
        if not isinstance(value, torch.Tensor):
            raise IdentityError(prefix+'.tensor_type.'+row['name'], 'Tensor', str(type(value)))
        for field, actual in [('shape',list(value.shape)), ('dtype',str(value.dtype)),
                              ('numel',value.numel()), ('bytes',value.numel()*value.element_size()),
                              ('sha256',raw_sha(value))]:
            require(prefix+'.'+row['name']+'.'+field, row[field], actual)
    require(prefix+'.state_metadata', identity['state_dict_metadata'], dict(getattr(state,'_metadata',{})))


def load_into(model, identity=None):
    """No new RNG seed/draw, optimizer, architecture call or parameter alias assignment."""
    identity = identity or json.loads(safe('initialization/IDENTITY.json').read_text())
    require('weight_role', 'ZERO_STEP_UNTRAINED_NOT_PRETRAINED', identity['role'])
    path = safe(identity['file'])
    require('zero_step_file.sha256', identity['file_sha256'], sha(path))
    require('zero_step_file.size', identity['file_bytes'], path.stat().st_size)
    require('architecture.sha256', identity['source_architecture_sha256'], sha(safe('vendor/RetinexFormer_arch.py')))
    before_rng = rng_identity()
    # Only this explicit new zero-step file is ever loaded. No fallback/history path.
    state = torch.load(path, map_location='cpu', weights_only=True)
    verify_state_mapping(state, identity, 'serialized_state')
    require('model_schema', [(r['name'],r['role'],r['shape'],r['dtype']) for r in identity['tensors']],
            [(r['name'],r['role'],r['shape'],r['dtype']) for r in inventory(model)])
    try:
        result = model.load_state_dict(state, strict=True)  # copy, not assign; independent storage
    except Exception as error:
        raise IdentityError('strict_load', 'missing=0 unexpected=0 and compatible shapes', str(error)) from error
    require('strict_load.missing', [], list(result.missing_keys))
    require('strict_load.unexpected', [], list(result.unexpected_keys))
    verify_state_mapping(model.state_dict(), identity, 'loaded_model')
    require('all_parameters_and_buffers', identity['tensors'], inventory(model))
    require('loaded_legacy_initializer_sha256', identity['legacy_initializer_sha256'], legacy_sha(model))
    require('load_preserves_post_constructor_rng', before_rng, rng_identity())
    return {'status':'PASS', 'file_sha256':identity['file_sha256'], 'state_tensors':len(state),
            'parameter_tensors':identity['parameter_tensors'], 'buffer_tensors':identity['buffer_tensors'],
            'legacy_initializer_sha256':legacy_sha(model), 'missing':0, 'unexpected':0,
            'RNG_unchanged_by_load':True, 'post_constructor_rng':before_rng}


def sealed_scratch_cpu(emit=None):
    identity = json.loads(safe('initialization/IDENTITY.json').read_text())
    try:
        model = scratch_cpu()  # unchanged V1 seed calls and full constructor RNG consumption
    except Exception as error:
        failure = IdentityError('onsite_constructor', identity['legacy_initializer_sha256'],
                                {'digest':'UNAVAILABLE','error':repr(error)})
        if emit: emit({'event':'INITIALIZER_CONSTRUCTION_FAILURE',**failure.detail})
        raise failure from error
    actual = legacy_sha(model)
    fresh = {'check':'onsite_fresh_initializer_vs_V1', 'expected':identity['legacy_initializer_sha256'],
             'actual':actual, 'status':'MATCH' if actual==identity['legacy_initializer_sha256'] else 'MISMATCH',
             'blocking_in_V2':False, 'historical_RUN01':'FAIL_actual_not_recorded_cause_unresolved'}
    if emit: emit({'event':'FRESH_INITIALIZER_DIAGNOSTIC', **fresh})
    try:
        loaded = load_into(model, identity)
    except Exception as error:
        if emit:
            emit({'event':'SEALED_INITIALIZER_FAILURE',
                  **getattr(error,'detail',{'check':'sealed_load','expected':'exact identity','actual':repr(error)})})
        raise
    if emit: emit({'event':'SEALED_INITIALIZER_PASS', **loaded})
    return model, {'fresh_initialization':fresh, 'sealed_load':loaded}
