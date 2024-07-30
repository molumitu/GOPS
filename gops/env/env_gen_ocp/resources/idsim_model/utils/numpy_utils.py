import numpy as np
from typing import Literal, Tuple, overload


# def readonly_array(*args, **kwargs) -> np.ndarray:
#     arr: np.ndarray = np.array(*args, **kwargs)
#     arr.flags.writeable = False
#     return arr

@overload
def pad_with_mask(arr: np.ndarray, max_len: int, ego_state: np.ndarray, sur_obs_padding: str, padding_shape: np.ndarray, separate_mask: Literal[False] = ...) -> np.ndarray: ...
@overload
def pad_with_mask(arr: np.ndarray, max_len: int, ego_state: np.ndarray, sur_obs_padding: str, padding_shape: np.ndarray, separate_mask: Literal[True]) -> Tuple[np.ndarray, np.ndarray]: ...
def pad_with_mask(arr: np.ndarray, max_len: int, ego_state: np.ndarray = np.zeros(6), sur_obs_padding: str = 'zero', padding_shape: np.ndarray = np.zeros(2),  separate_mask: bool = False) -> np.ndarray:
    seq_dim, *rest_dims = arr.shape
    assert seq_dim <= max_len
    if separate_mask:
        arr2 = np.zeros((max_len, *rest_dims), dtype=arr.dtype)
        mask = np.zeros(max_len, dtype=np.bool8)
        arr2[:seq_dim] = arr
        arr2[seq_dim:] = padding_sur_obs(arr2[seq_dim:], ego_state, sur_obs_padding, padding_shape)
        mask[:seq_dim] = True

        return arr2, mask
    else:
        assert len(rest_dims) == 1
        feature_dim = rest_dims[0]
        arr2 = np.zeros((max_len, feature_dim + 1), dtype=arr.dtype)
        arr2[:seq_dim, :feature_dim] = arr
        arr2[:seq_dim, feature_dim] = 1.0
        arr2[seq_dim:, :feature_dim] = padding_sur_obs(arr2[seq_dim:, :feature_dim], ego_state, sur_obs_padding, padding_shape)
        return arr2
    
def padding_sur_obs(sur_obs: np.ndarray, ego_state: np.ndarray, sur_obs_padding: str, padding_shape: np.ndarray) -> np.ndarray:
    padding_num = sur_obs.shape[0]
    if padding_num == 0:
        pad_obs = sur_obs
    else:
        ego_partial_state = np.array([ego_state[0], ego_state[1], ego_state[4], ego_state[2]], dtype=np.float32) # (x, y, phi, speed) 
        if sur_obs_padding == 'zero': # NOTE: not strictly correct, depends on the corrodinate system
            base_padding = np.concatenate((ego_partial_state, np.zeros(2))).reshape(1, -1)
            pad_obs = base_padding.repeat(padding_num, axis=0)

        elif sur_obs_padding == 'rule':
            bias = np.array([-30, 0, 0, 0], dtype=np.float32)
            sur_state = ego_partial_state + bias
            padding_base = np.concatenate((sur_state, padding_shape)).reshape(1, -1) 
            padding_interval = np.array([-10, 0, 0, 0, 0, 0], dtype=np.float32)
            pad_obs = padding_base + padding_interval * np.arange(padding_num).reshape(-1, 1)
        else: 
            raise RuntimeError(f'unknown sur_obs_padding: {sur_obs_padding}')
    return pad_obs

        
