from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration


def load_llm(model_name):
    if "Qwen3-VL" in model_name:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, dtype="auto", device_map="auto")
        processor = AutoProcessor.from_pretrained(model_name)
    elif "Qwen2.5-VL" in model_name:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        processor = AutoProcessor.from_pretrained(model_name)
    else:
        raise NotImplementedError
    return model, processor


def infer_llm(model, processor, messages, max_new_tokens=1024):
    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text
