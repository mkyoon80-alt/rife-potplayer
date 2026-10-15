from pathlib import Path
import argparse, hashlib
import onnx
from onnx import helper, TensorProto
parser=argparse.ArgumentParser(description="Extract the original-resolution RIFE motion/mask guide")
parser.add_argument("source",type=Path);parser.add_argument("output",type=Path);args=parser.parse_args()
assert hashlib.sha256(args.source.read_bytes()).hexdigest()=="65c57a5e4abb17ad67faf35054291ac53affab0506d509d2e66698f6ecd75584", "Wrong RIFE source export"
m=onnx.load(args.source)
m.graph.node.append(helper.make_node('Concat',['/Add_7_output_0','/Sigmoid_output_0'],['lab_padded_motion'],axis=1,name='lab_concat'))
m.graph.node.append(helper.make_node('Pad',['lab_padded_motion','/Cast_9_output_0',''],['motion'],name='lab_crop'))
del m.graph.output[:];m.graph.output.append(helper.make_tensor_value_info('motion',TensorProto.FLOAT,[1,5,'height','width']))
needed={'motion'};kept=[]
for node in reversed(m.graph.node):
    if any(x in needed for x in node.output):kept.append(node);needed.update(x for x in node.input if x)
del m.graph.node[:];m.graph.node.extend(reversed(kept))
initializers=[x for x in m.graph.initializer if x.name in needed]
del m.graph.initializer[:];m.graph.initializer.extend(initializers);del m.graph.value_info[:]
onnx.checker.check_model(m);onnx.save(m,args.output)
print(hashlib.sha256(args.output.read_bytes()).hexdigest())
