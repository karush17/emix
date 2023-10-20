"""Stores the episode buffer."""

from types import SimpleNamespace as SN

import torch as th
import numpy as np


class EpisodeBatch:
    """Implements a batch of episode samples.
    
    Attributes:
        scheme: scheme for storing data samples.
        groups: episode groups.
        batch_size: size of each batch of data samples.
        max_seq_length: length of the maximum sequence.
        preprocess: whether to preprocess data samples or not.
        device: device id used for training.
        data: data storage buffer.
    """
    def __init__(self,
                 scheme,
                 groups,
                 batch_size,
                 max_seq_length,
                 data=None,
                 preprocess=None,
                 device="cpu"):
        """Initialize the episode batch object."""
        self.scheme = scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        self.device = device

        if data is not None:
            self.data = data
        else:
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            self._setup_data(self.scheme, self.groups,
                             batch_size, max_seq_length, self.preprocess)

    def _setup_data(self, scheme, groups, batch_size,
                    max_seq_length, preprocess):
        """Setup the data samples.
        
        Args:
            scheme: buffer scheme.
            groups: episode groups.
            batch_size: size of each data batch.
            max_seq_length: length of the maximum sequence.
            preprocess: whether to preprocess samples or not.
        """
        if preprocess is not None:
            for k in preprocess:
                assert k in scheme
                new_k = preprocess[k][0]
                transforms = preprocess[k][1]

                vshape = self.scheme[k]["vshape"]
                dtype = self.scheme[k]["dtype"]
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                self.scheme[new_k] = {
                    "vshape": vshape,
                    "dtype": dtype
                }
                if "group" in self.scheme[k]:
                    self.scheme[new_k]["group"] = self.scheme[k]["group"]
                if "episode_const" in self.scheme[k]:
                    self.scheme[new_k]["episode_const"] = self.scheme[k]["episode_const"]

        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {"vshape": (1,), "dtype": th.long},
        })

        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", th.float32)

            if isinstance(vshape, int):
                vshape = (vshape,)

            if group:
                assert group in groups, "Group {} must have its number of members defined in _groups_".format(group)
                shape = (groups[group], *vshape)
            else:
                shape = vshape

            if episode_const:
                self.data.episode_data[field_key] = th.zeros((batch_size, *shape),
                                                             dtype=dtype,
                                                             device=self.device)
            else:
                self.data.transition_data[field_key] = th.zeros((batch_size,
                                                                 max_seq_length,
                                                                 *shape),
                                                                 dtype=dtype,
                                                                 device=self.device)

    def extend(self, scheme, groups=None):
        """Extend the episode buffer.
        
        Args:
            scheme: buffer scheme.
            groups: epsiode groups.
        """
        self._setup_data(scheme,
                         self.groups if groups is None else groups,
                         self.batch_size, self.max_seq_length)

    def to(self, device):
        """Move tensor to device.
        
        Args:
            device: device id of the new device.
        """
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        """Update the episode batch.
        
        Args:
            data: new data samples.
            bs: batch size.
            ts: tensor size.
            mark_filled: buffer filled till last mark or not.
        """
        slices = self._parse_slices((bs, ts))
        for k, v in data.items():
            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled:
                    target["filled"][slices] = 1
                    mark_filled = False
                _slices = slices
            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            dtype = self.scheme[k].get("dtype", th.float32)
            v = th.tensor(v, dtype=dtype, device=self.device)
            self._check_safe_view(v, target[k][_slices])
            target[k][_slices] = v.view_as(target[k][_slices])

            if k in self.preprocess:
                new_k = self.preprocess[k][0]
                v = target[k][_slices]
                for transform in self.preprocess[k][1]:
                    v = transform.transform(v)
                target[new_k][_slices] = v.view_as(target[new_k][_slices])

    def _check_safe_view(self, v, dest):
        """Assert view of tensor.
        
        Args:
            v: input tensor.
            dest: desired view.
        """
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError("Unsafe reshape of {} to {}".format(v.shape, dest.shape))
            else:
                idx -= 1

    def __getitem__(self, item):
        """Retrive an item from the dictionary.
        
        Args:
            item: dictionary value.
        """
        if isinstance(item, str):
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]]
                          for key in item if "group" in self.scheme[key]}
            ret = EpisodeBatch(new_scheme, new_groups, self.batch_size,
                               self.max_seq_length,
                               data=new_data, device=self.device)
            return ret
        else:
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size)
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)

            ret = EpisodeBatch(self.scheme, self.groups, ret_bs, ret_max_t,
                               data=new_data, device=self.device)
            return ret

    def _get_num_items(self, indexing_item, max_size):
        """Retrieve numbered items.
        
        Args:
            indexing_item: item index.
            max_size: maximum size of the collection.
        """
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1)//_range[2]

    def _new_data_sn(self):
        """Initialize new data storage."""
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        """Parse tensor slices.
        
        Args:
            items: data collection items.
        """
        parsed = []
        # Only batch slice given, add full time slice
        if (isinstance(items, slice)  # slice a:b
            or isinstance(items, int)  # int i
            or (isinstance(items, (list, np.ndarray, th.LongTensor,
                                   th.cuda.LongTensor)))  # [a,b,c]
            ):
            items = (items, slice(None))

        # Need the time indexing to be contiguous
        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous")

        for item in items:
            #TODO: stronger checks to ensure only supported options get through
            if isinstance(item, int):
                # Convert single indices to slices
                parsed.append(slice(item, item+1))
            else:
                # Leave slices and lists as is
                parsed.append(item)
        return parsed

    def max_t_filled(self):
        """Maximum tensor filled."""
        return th.sum(self.data.transition_data["filled"], 1).max(0)[0]

    def __repr__(self):
        """Batch data information."""
        return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{} Groups:{}".format(self.batch_size,
                                                                                     self.max_seq_length,
                                                                                     self.scheme.keys(),
                                                                                     self.groups.keys())


class ReplayBuffer(EpisodeBatch):
    """Initializes the Agent Replay Buffer.
    
    Attributes:
        buffer_size: size of the replay buffer.
        buffer_index: current position in the buffer.
        episodes_in_buffer: count of episodes stored in buffer.
    """
    def __init__(self, scheme, groups, buffer_size,
                 max_seq_length, preprocess=None, device="cpu"):
        """Initializes the buffer object."""
        super(ReplayBuffer, self).__init__(scheme, groups,
                                           buffer_size,
                                           max_seq_length,
                                           preprocess=preprocess,
                                           device=device)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0

    def insert_episode_batch(self, ep_batch):
        """Inserts a batch of episodes.
        
        Args:
            ep_batch: batch of episode samples.
        """
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index,
                              self.buffer_index + ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index,
                              self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer,
                                          self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def can_sample(self, batch_size):
        """Whether it is possible to sample from buffer or not.
        
        Args:
            batch_size: size of each batch.
        """
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size):
        """Sample data from replay buffer.
        
        Args:
            batch_size: size of each batch.
        """
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(self.episodes_in_buffer,
                                      batch_size, replace=False)
            return self[ep_ids]

    def __repr__(self):
        """Return replay buffer information."""
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.episodes_in_buffer,
                                                                        self.buffer_size,
                                                                        self.scheme.keys(),
                                                                        self.groups.keys())
