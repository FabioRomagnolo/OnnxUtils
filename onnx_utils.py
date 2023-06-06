import torch
import onnx
import onnxruntime

from onnx import helper as h
from onnx import checker as ch
from onnx import TensorProto
from onnx import numpy_helper as nph
from onnxsim import simplify

import time
import os
import numpy as np
from collections import OrderedDict


def make_param_dictionary(initializer):
    params = OrderedDict()
    for data in initializer:
        params[data.name] = data
    return params


def convert_params_to_int32(params_dict):
    converted_params = []
    for param in params_dict:
        data = params_dict[param]
        if data.data_type == TensorProto.INT64:
            data_cvt = nph.to_array(data).astype(np.int32)
            data = nph.from_array(data_cvt, data.name)
        converted_params += [data]
    return converted_params


def convert_constant_nodes_to_int32(nodes):
    """
    convert_constant_nodes_to_int32 Convert Constant nodes to INT32. If a constant node has data type INT64, a new version of the
    node is created with INT32 data type and stored.
    Args:
        nodes (list): list of nodes
    Returns:
        list: list of new nodes all with INT32 constants.
    """
    new_nodes = []
    for node in nodes:
        if node.op_type == "Constant":
            if node.attribute[0].t.data_type == TensorProto.INT64:
                data = nph.to_array(node.attribute[0].t).astype(np.int32)
                new_t = onnx.helper.make_tensor(
                    name='const_tensor_int32',
                    data_type=onnx.TensorProto.INT32,
                    dims=data.shape,
                    vals=data.flatten(),
                )
                new_node = h.make_node(
                    "Constant",
                    inputs=[],
                    outputs=node.output,
                    name=node.name,
                    value=new_t,
                )
                new_nodes += [new_node]
                continue

        new_nodes += [node]

    return new_nodes


def convert_model_to_int32(model_path: str, out_path: str, verbose=True):
    """
    convert_model_to_int32 Converts ONNX model with INT64 params to INT32 params.\n
    Args:\n
        model_path (str): path to original ONNX model.\n
        out_path (str): path to save converted model.
        verbose (bool)
    """
    if verbose:
        print(f"Loading Model: {model_path}")
    # * load model.
    model = onnx.load_model(model_path)
    ch.check_model(model)
    # * get model opset version.
    opset_version = model.opset_import[0].version
    graph = model.graph
    # * The initializer holds all non-constant weights.
    init = graph.initializer
    # * collect model params in a dictionary.
    params_dict = make_param_dictionary(init)
    if verbose:
        print("Converting INT64 model params to INT32...")
    # * convert all INT64 aprams to INT32.
    converted_params = convert_params_to_int32(params_dict)
    if verbose:
        print("Converting constant INT64 nodes to INT32...")
    new_nodes = convert_constant_nodes_to_int32(graph.node)

    graph_name = f"{graph.name}-int32"
    if verbose:
        print("Creating new graph...")
    # * create a new graph with converted params and new nodes.
    graph_int32 = h.make_graph(
        new_nodes,
        graph_name,
        graph.input,
        graph.output,
        initializer=converted_params,
    )
    if verbose:
        print("Creating new int32 model...")
    model_int32 = h.make_model(graph_int32, producer_name="onnx-typecast")
    model_int32.opset_import[0].version = opset_version
    ch.check_model(model_int32)
    if verbose:
        print(f"Saving converted model as: {out_path}")
    onnx.save_model(model_int32, out_path)
    if verbose:
        print(f"--- int32 conversion completed.")
    return


def replace_constantofshape_nodes(nodes, verbose=True):
    idxs_tofix = []
    for i in range(len(nodes)):
        if nodes[i].op_type == "ConstantOfShape":
            idxs_tofix.append(i)

    # Replacing the nodes
    for i in idxs_tofix:
        old_constantofshape = nodes[i]
        old_name = old_constantofshape.name.split('_')[0]
        old_outputs = old_constantofshape.output

        # If input of constantofshape is empty, then a scalar will be given!
        old_input = []
        if len(old_constantofshape.input) > 0:
            old_input = old_constantofshape.input[0]

        if len(old_constantofshape.attribute) > 0:
            if old_constantofshape.attribute[0].t.data_type != TensorProto.UNDEFINED:
                old_value = old_constantofshape.attribute[0].t
            else:
                # If a value is not provided for ConstantOfShape, a 0 tensor must be created!
                old_value = create_zero_tensor()
        else:
            # If a value is not provided for ConstantOfShape, a 0 tensor must be created!
            old_value = create_zero_tensor()

        # If input of constantofshape is empty, then a scalar will be given!
        # Constant node can give scalar output specifying float32 or int32 value as output
        if len(old_input) == 0:
            # old_value = np.asscalar(np.frombuffer(old_value.raw_data, dtype=np.int32))
            old_value = old_value.int32_data[0]
            new_constant = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=old_outputs,
                name=f"{old_constantofshape.name}".replace(old_name, 'Constant'),
                value_float=float(old_value)     # Value for the scalar
            )
            # Update node
            nodes[i] = new_constant
        else:
            new_constant = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=[f'Tile_{i + 1}'],
                name=f"{old_constantofshape.name}".replace(old_name, 'Constant'),
                value=old_value  # Value for the constant tensor
            )
            new_tile = onnx.helper.make_node(
                'Tile',
                inputs=[f'Tile_{i+1}', old_input],   # Inputs: [tensor, repeats_tensor (it's the shape!)]
                outputs=old_outputs,
                name=f"{new_constant.name}".replace('Constant', 'Tile'),
            )
            # Update nodes
            nodes[i] = new_constant
            nodes.insert(i + 1, new_tile)
            # WARNING! The idxs must be updated!
            for c, idx in enumerate(idxs_tofix):
                idxs_tofix[c] += 1

    if verbose:
        print(f"{len(idxs_tofix)} ConstantOfShape fixed!")
    return nodes


def create_zero_tensor():
    zero_np = np.zeros([1]).astype(np.int32)
    return onnx.helper.make_tensor(
        name='const_zero_tensor',
        data_type=onnx.TensorProto.INT32,
        dims=zero_np.shape,
        vals=zero_np.flatten(),
    )


def get_onnx_operators(model_path: str, verbose=True):
    operators = {}
    # * load model.
    model = onnx.load_model(model_path)
    ch.check_model(model)
    graph = model.graph
    # * iterating over nodes
    for node in graph.node:
        if operators.get(node.op_type, None) is None:
            operators[node.op_type] = 1
        else:
            operators[node.op_type] += 1
    # * showing operators
    if verbose:
        print(f"\n--- {len(operators.keys())} ONNX operators involved in {model_path}:")
        for k in sorted(operators.keys()):
            print(f"- {k}: appears {operators[k]} times")
        print("--- End of ONNX operators\n")
    return operators


def save_new_model(opset_version, nodes, graph, out_path, verbose=True):
    if verbose:
        print("Creating new fixed graph...")
    # * create a new graph with new nodes.
    new_graph = h.make_graph(
        nodes,
        graph.name,
        graph.input,
        graph.output,
        initializer=graph.initializer,  # The initializer holds all non-constant weights.
    )
    if verbose:
        print("Creating new fixed model...")
    new_model = h.make_model(new_graph, producer_name="onnx-fix-nodes")
    new_model.opset_import[0].version = opset_version
    ch.check_model(new_model)
    if verbose:
        print(f"Saving new model as: {out_path}")
    onnx.save_model(new_model, out_path)


def fix_onnx_resize_nodes(model_path: str, out_path: str, verbose=True):
    """
    Method to fix resize nodes giving the following error in Tensorflow
    conversions:
    - "Resize coordinate_transformation_mode=pytorch_half_pixel is not supported in Tensorflow"
    """
    if verbose:
        print(f"Loading Model: {model_path}")
    # * load model.
    model = onnx.load_model(model_path)
    ch.check_model(model)
    # * get model opset version.
    opset_version = model.opset_import[0].version
    graph = model.graph

    new_nodes = []
    if verbose:
        print("Fixing Resize nodes...")
    for i, node in enumerate(graph.node):
        if node.op_type == "Resize":
            new_resize = onnx.helper.make_node(
                'Resize',
                inputs=node.input,
                outputs=node.output,
                name=node.name,
                coordinate_transformation_mode='half_pixel',  # Instead of pytorch_half_pixel, unsupported by Tensorflow
                mode='linear',
            )
            # Update node
            new_nodes += [new_resize]
        else:
            new_nodes += [node]

    save_new_model(opset_version=opset_version, graph=graph, nodes=new_nodes,
                   out_path=out_path, verbose=verbose)


def validate_onnx(onnx_file, pytorch_model, src, bgr, device='cuda', warnings=False):
    print(f'- Validating ONNX model...')
    if warnings:
        onnxruntime.set_default_logger_severity(0)

    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)

    with torch.no_grad():
        start_time = time.time()
        out_torch = pytorch_model(src, bgr)
        pytorch_inference_seconds = time.time() - start_time
    print("PyTorch inference seconds: ", pytorch_inference_seconds)

    src_numpy = src.cpu().numpy()
    bgr_numpy = bgr.cpu().numpy()

    if device == 'cuda':
        providers = [
            'CUDAExecutionProvider',  # 1st choice
            'CPUExecutionProvider'    # 2nd choice
        ]
    else:
        providers = ['CPUExecutionProvider']

    sess = onnxruntime.InferenceSession(onnx_file, providers=providers)
    output_names = [o.name for o in sess.get_outputs()]

    if device == 'cuda':
        io_binding = sess.io_binding()
        # OnnxRuntime will copy the data over to the CUDA device if 'input' is consumed by nodes on the CUDA device
        io_binding.bind_cpu_input('src', src_numpy)
        io_binding.bind_cpu_input('bgr', bgr_numpy)
        for o in output_names:
            io_binding.bind_output(o)

        start_time = time.time()
        sess.run_with_iobinding(io_binding)
        onnx_inference_seconds = time.time() - start_time

        out_onnx = io_binding.copy_outputs_to_cpu()
    else:
        start_time = time.time()
        out_onnx = sess.run(None, {
            'src': src_numpy,
            'bgr': bgr_numpy
        })
        onnx_inference_seconds = time.time() - start_time

    print("ONNX inference seconds: ", onnx_inference_seconds)
    print(f"Inference seconds differs by {pytorch_inference_seconds - onnx_inference_seconds}")
    e_max = 0
    for a, b, name in zip(out_torch, out_onnx, output_names):
        b = torch.as_tensor(b)
        e = torch.abs(a.cpu() - b).max()
        e_max = max(e_max, e.item())
        print(f'"{name}" output differs by maximum of {e}')

    if e_max < 0.005:
        print('ONNX Validation passed.')
    else:
        print("WARNING: ONNX validation failed.")


def simplify_onnx_model(path_to_onnx, model_name, dummy_src, dummy_bgr,
                        convert_to_int32=False, verbose=True):
    if verbose:
        print("- Simplifying the ONNX model ...")
    onnx_simplified, check = simplify(
        onnx.load(path_to_onnx),
        dynamic_input_shape=False,
        input_shapes={
            'src': list(dummy_src.shape), 'bgr': list(dummy_bgr.shape)
        },
        skipped_optimizers=[]
    )
    if check:
        # Validation passed
        path_to_onnx_simplified = os.path.join(os.path.dirname(path_to_onnx), f"onnx_{model_name}_simplified.onnx")
        onnx.save(onnx_simplified, path_to_onnx_simplified)
        old_onnx_operators = get_onnx_operators(path_to_onnx, verbose=False)
        new_onnx_operators = get_onnx_operators(path_to_onnx_simplified, verbose=True)
        if verbose:
            for k, v in old_onnx_operators.items():
                if k not in new_onnx_operators.keys():
                    print(f"- Op removed in simplified model: {k}")
            for k, v in new_onnx_operators.items():
                if k not in old_onnx_operators.keys():
                    print(f"- New op added in simplified model: {k} appearing {v} times")
            print(f"--- The optimization removed {len(old_onnx_operators) - len(new_onnx_operators)} operators!")

        if convert_to_int32:
            if verbose:
                print("--- Converting to int32 all the ONNX model params in the simplified model ...")
                # TODO: FIX "graph attribute is not supported yet"
            convert_model_to_int32(path_to_onnx_simplified, path_to_onnx_simplified, verbose=verbose)
        return path_to_onnx_simplified
    else:
        # Validation failed
        print("WARNING: Simplified ONNX model could not be validated, returning empty path ...")
        return None
