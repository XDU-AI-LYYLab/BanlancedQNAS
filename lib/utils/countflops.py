import numpy as np
import torch
from tabulate import tabulate
from torch import nn


class FLOPS_Counter:
    def __init__(self, model, input_shape, arch_param=False):
        self.input_shape = input_shape
        self.total_gflops = 0
        model.eval()
        backend = HookBackend(arch_param=arch_param)
        self.layers = backend(model, input_shape)
        self.process_layers()

    def print_layers(self):
        ignored_layers = {'Relu', 'BatchNorm2d'}
        print('Ignoring the following layers:', ignored_layers)
        layers = [layer for layer in self.layers if layer['attributes']
                  ['name'] not in ignored_layers]
        print(
            _tabulate(
                layers,
                ('attributes',
                 '% gflops',
                 'gflops',
                 'nbytes',
                 'in_shape',
                 'out_shape')))
        return layers

    def print_summary(self, print_flag=False):
        total_dict = {
            'total_gflops': self.total_gflops,
            'resolution': self.input_shape,
        }
        if print_flag:
            print(_tabulate([total_dict], total_dict.keys()))
        return total_dict

    def process_layers(self):
        for layer in self.layers:
            # shape is BxCxHxW
            if layer['attributes']['name'] == 'Conv2d':
                kernel = layer['attributes']['kernel']
                layer['gflops'] = (np.prod(layer['out_shape']) *
                                   np.prod(layer['attributes']['kernel']) *
                                   layer['in_shape'][1] /
                                   (10 ** 9 *
                                    layer['attributes']['groups']))
                self.total_gflops += layer['gflops']

            layer['in_shape'] = 'x'.join(map(str, layer['in_shape']))
            layer['out_shape'] = 'x'.join(map(str, layer['out_shape']))

        for layer in self.layers:
            layer['% gflops'] = str(
                round(
                    layer.get(
                        'gflops',
                        0) / self.total_gflops * 100,
                    3)) + '%'


class HookBackend:
    def __init__(self, arch_param=False):
        self.arch_param = arch_param
        self.layers = []

    def __call__(self, model, input_shape):
        def hook(module, input, output):
            if list(module.children()) != []:
                return  # ignore modules with children, e.g. nn.Sequential
            assert len(input) == 1
            input = input[0]
            attributes = self.get_attributes(module)
            self.layers.append({
                'attributes': attributes,
                'in_shape': input.shape,
                'out_shape': output.shape,
            })

        def register_hook(module):
            handle = module.register_forward_hook(hook)
            handles.append(handle)

        handles = []
        model.apply(register_hook)

        data = torch.rand(
            *
            input_shape).to(
            next(
                model.parameters()).device)  # get data on same device as model

        with torch.no_grad():
            model(data) if not self.arch_param else model(
                data, self.arch_param)
        for handle in handles:
            handle.remove()

        return self.layers

    def get_attributes(self, module):
        if isinstance(module, nn.Conv2d):
            return {
                'name': 'Conv2d',
                'kernel': module.kernel_size,
                'groups': module.groups,
                'stride': module.stride,
            }
        if isinstance(module, nn.ReLU):
            return {
                'name': 'Relu',
            }
        if isinstance(module, nn.BatchNorm2d):
            return {
                'name': 'BatchNorm2d',
            }
        else:
            return {
                'name': str(module),
            }


def _tabulate(data, keys):
    """ Tabulate does not allow printing only specific keys from dictionaries,
    so this helper function takes a list of dicts and creates a list of lists, where each
    list is the corresponding dict, with only the provided keys (in the provided order).
    This allows easy printing of large dicts where only specificed keys are printed
    """
    new_data = []
    for layer in data:
        new_data.append([layer.get(key, None) for key in keys])

    return tabulate(new_data, headers=keys, tablefmt='fancy_grid')
