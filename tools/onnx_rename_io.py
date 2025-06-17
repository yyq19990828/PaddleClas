import argparse
import onnx
import onnx_graphsurgeon as gs

def rename_onnx_io(onnx_path, output_path, input_names=None, output_names=None):
    graph = gs.import_onnx(onnx.load(onnx_path))

    # 修改输入名
    if input_names:
        assert len(input_names) == len(graph.inputs), "输入数量与模型不符"
        for i, new_name in enumerate(input_names):
            old_name = graph.inputs[i].name
            graph.inputs[i].name = new_name
            # 同步修改所有引用
            for node in graph.nodes:
                node.inputs = [graph.inputs[i] if inp.name == old_name else inp for inp in node.inputs]

    # 修改输出名
    if output_names:
        assert len(output_names) == len(graph.outputs), "输出数量与模型不符"
        for i, new_name in enumerate(output_names):
            old_name = graph.outputs[i].name
            graph.outputs[i].name = new_name
            # 同步修改所有引用
            for node in graph.nodes:
                node.outputs = [graph.outputs[i] if out.name == old_name else out for out in node.outputs]

    onnx.save(gs.export_onnx(graph), output_path)
    print(f"已保存修改后的ONNX模型到: {output_path}")

def add_onnx_metadata(onnx_path, output_path, metadata_dict):
    model = onnx.load(onnx_path)
    # 清除同名元数据，避免重复
    existing_keys = set()
    for prop in model.metadata_props:
        existing_keys.add(prop.key)
    for k, v in metadata_dict.items():
        if k in existing_keys:
            # 更新已存在的key
            for prop in model.metadata_props:
                if prop.key == k:
                    prop.value = v
        else:
            meta = model.metadata_props.add()
            meta.key = k
            meta.value = v
    onnx.save(model, output_path)
    print(f"已保存添加元数据后的ONNX模型到: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX输入输出名重命名工具 (GraphSurgeon版)")
    parser.add_argument("--onnx", required=True, help="原始ONNX模型路径")
    parser.add_argument("--output", required=True, help="输出ONNX模型路径")
    parser.add_argument("--input_names", nargs='*', help="新的输入名（空格分隔，顺序与原模型一致）")
    parser.add_argument("--output_names", nargs='*', help="新的输出名（空格分隔，顺序与原模型一致）")
    parser.add_argument("--add_metadata", nargs='+', help="添加元数据，格式为 key=value（可多个，空格分隔，值可为带中括号的字符串）")
    parser.add_argument("--enable_metadata", action="store_true", help="是否启用元数据添加功能")
    args = parser.parse_args()

    rename_onnx_io(args.onnx, args.output, args.input_names, args.output_names)

    # 添加元数据（如果有且启用）
    if args.enable_metadata and args.add_metadata:
        import re
        meta_dict = {}
        # 拼接所有 add_metadata 参数为一个字符串再分割
        joined = ' '.join(args.add_metadata)
        # 支持多个 key=[...] 或 key="..." 或 key='...' 或 key=xxx
        pattern = r'(\w+)\s*=\s*(\[[^\]]*\]|"[^"]*"|\'[^\']*\'|[^\s]+)'
        for match in re.finditer(pattern, joined):
            k, v = match.group(1), match.group(2)
            meta_dict[k] = v
        add_onnx_metadata(args.output, args.output, meta_dict)
