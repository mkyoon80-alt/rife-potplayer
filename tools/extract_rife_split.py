"""Extract the unchanged pair-local Head boundary; developer-only ONNX tool."""
import argparse
import hashlib
from pathlib import Path
import onnx
from onnx import helper, numpy_helper

parser = argparse.ArgumentParser()
parser.add_argument('motion_model', type=Path)
parser.add_argument('output_directory', type=Path)
args = parser.parse_args()
assert hashlib.sha256(args.motion_model.read_bytes()).hexdigest() == '95c59f2b45ab358981e7ec044b946abd69960e29d034bc16e99223768b284506'
model = onnx.load(args.motion_model)
nodes = {n.name: n for n in model.graph.node}
split = nodes['/Split']
assert helper.get_attribute_value(split.attribute[0]) == 1
sizes = next(n for n in model.graph.node if split.input[1] in n.output)
assert numpy_helper.to_array(next(a.t for a in sizes.attribute if a.name == 'value')).tolist() == [6, 1]
assert nodes['/Reshape_11'].input[0] == split.output[0]
assert nodes['/encode/cnn0/Conv'].input[0] == nodes['/Reshape_11'].output[0]
assert next(a.s for a in nodes['/Pad'].attribute if a.name == 'mode') == b'reflect'
for value in model.graph.input:
    if value.name == 'input':
        for dimension, size in zip(value.type.tensor_type.shape.dim, [1, 7, 540, 960]):
            dimension.dim_value = size
boundary = '/encode/cnn3/ConvTranspose_output_0'
extractor = onnx.utils.Extractor(onnx.shape_inference.infer_shapes(model, strict_mode=True))
head = extractor.extract_model(['input'], [boundary])
tail = extractor.extract_model(['input', boundary], ['motion'])
args.output_directory.mkdir(parents=True, exist_ok=True)
for part, name in [(head, 'head'), (tail, 'tail')]:
    for node in part.graph.node:
        for values in [node.input, node.output]:
            for i, value in enumerate(values):
                if value == boundary:
                    values[i] = 'features'
    for values in [part.graph.input, part.graph.output, part.graph.value_info]:
        for value in values:
            if value.name == boundary:
                value.name = 'features'
            if value.name == 'features':
                value.type.tensor_type.CopyFrom(helper.make_tensor_value_info('features', onnx.TensorProto.FLOAT, [2, 4, 544, 960]).type.tensor_type)
    onnx.checker.check_model(part, full_check=True)
    destination = args.output_directory / f'rife425_{name}.onnx'
    onnx.save(part, destination)
    print(hashlib.sha256(destination.read_bytes()).hexdigest(), destination.name)
