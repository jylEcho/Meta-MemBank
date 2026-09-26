import torch.nn as nn

# 代码复制自：https://github.com/huggingface/peft/blob/2f5360a7da22a236b5ad4c059572fff5321c867c/src/peft/peft_model.py#L617
def get_nb_trainable_parameters(model:nn.Module) -> tuple[int, int]:
    """
    返回模型中可训练参数的数量和所有参数的数量。

    参数:
        model (nn.Module): 要统计参数的模型

    返回:
        tuple[int, int]: 一个元组，包含可训练参数的数量和所有参数的数量
    """
    # 初始化可训练参数的数量
    trainable_params = 0
    # 初始化所有参数的数量
    all_param = 0
    # 遍历模型的所有命名参数
    for _, param in model.named_parameters():
        # 获取当前参数的元素数量
        num_params = param.numel()
        # 如果使用 DeepSpeed Zero 3 并且权重初始化为空
        if num_params == 0 and hasattr(param, "ds_numel"):
            # 使用 DeepSpeed 统计的元素数量
            num_params = param.ds_numel

        # 由于 bitsandbytes 库中 4 位线性层的设计
        # 需要将参数数量乘以 2 以获得正确的参数数量
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "element_size"):
                # 获取元素的字节大小
                num_bytes = param.element_size()
            elif not hasattr(param, "quant_storage"):
                # 如果没有量化存储属性，默认字节大小为 1
                num_bytes = 1
            else:
                # 获取量化存储的字节大小
                num_bytes = param.quant_storage.itemsize
            # 调整参数数量
            num_params = num_params * 2 * num_bytes

        # 累加所有参数的数量
        all_param += num_params
        # 如果当前参数需要梯度更新
        if param.requires_grad:
            # 累加可训练参数的数量
            trainable_params += num_params

    return trainable_params, all_param


# 代码复制自：https://github.com/huggingface/peft/blob/2f5360a7da22a236b5ad4c059572fff5321c867c/src/peft/peft_model.py#L647
def print_trainable_parameters(model: nn.Module) -> None:
    """
    打印模型中可训练参数的数量。

    注意：print_trainable_parameters() 使用了 get_nb_trainable_parameters()，这与 huggingface/transformers 中的
    num_parameters(only_trainable=True) 不同。get_nb_trainable_parameters() 返回的是 Peft 模型的
    （可训练参数，所有参数），其中包括修改后的骨干变压器模型。对于像 LoRA 这样的技术，骨干变压器模型会被 LoRA 模块就地修改。
    然而，对于提示调优，骨干变压器模型不会被修改。num_parameters(only_trainable=True) 返回的是骨干变压器模型的
    可训练参数数量，这可能会有所不同。

    参数:
        model (nn.Module): 要打印可训练参数信息的模型
    """
    # 获取可训练参数和所有参数的数量
    trainable_params, all_param = get_nb_trainable_parameters(model)

    # 打印可训练参数数量、所有参数数量以及可训练参数的百分比
    print(
        f"可训练参数: {trainable_params:,d} || 所有参数: {all_param:,d} || 可训练百分比: {100 * trainable_params / all_param:.4f}"
    )


if __name__ == "__main__":
    from transformers import LlavaForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    import torch

    # 加载LLaVA模型
    model_name_or_path = "./external/data/private/pc/Tdebug/llava/train_my/checkpoint/my_llava"
    model = LlavaForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        torch_dtype=torch.bfloat16,  # 使用BF16精度
        low_cpu_mem_usage=True,  # 优化CPU内存使用
        local_files_only=True   # 仅使用本地文件
    )

    # LoRA参数配置
    LORA_R = 32  # 秩参数，控制低秩近似的维度
    # LORA_ALPHA = 16  # 缩放因子，用于调整 LoRA 模块中权重更新的幅度
    LORA_DROPOUT = 0.05  # dropout率
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # 需要应用LoRA的模块名称

    # 初始化LoRA配置
    config = LoraConfig(
        r=LORA_R,
        # lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",  # LoRA 只作用于权重矩阵，而不影响偏置项
        task_type="CAUSAL_LM",  # 表示任务类型是因果语言模型（Causal Language Modeling），即根据前面的文本预测下一个单词
        modules_to_save=["multi_modal_projector"],  # 表示在保存模型时，除了 LoRA 相关的参数外，还需要保存名称为 multi_modal_projector 的模块的参数
    )

    # 应用LoRA到模型
    model = get_peft_model(model, config)

    # 打印可训练参数信息
    print_trainable_parameters(model)
