import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2TokenizerFast


def generate_augmented_prompt(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast, initial_prompt, device: torch.device
):
    instruction_prompt = f'I have a short image caption. I want you to expand it to some more detailed captions. Use two sentences for each new caption. Try to use creative language and unique or distinct words that relate to the subject of interest. Generate 3 new captions, each one interpreting the original caption in a unique way i.e. each caption should clearly describe a different image. The captions do not need to be realistic, they are allowed to describe weird or unusual scenes. The short caption is: "{initial_prompt}"'
    input_ids = tokenizer.encode(instruction_prompt, return_tensors="pt")
    assert isinstance(input_ids, torch.Tensor)
    input_ids = input_ids.to(device=device)

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        # max_length=100,
        max_new_tokens=200,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]

    print(f"Original Prompt: '{initial_prompt}'")
    print(f"Generated text: '{gen_text}'")
    print("\n\n")


def main():
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16)
    model = model.to(device=device)
    assert isinstance(model, GPT2LMHeadModel)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    assert isinstance(tokenizer, GPT2TokenizerFast)

    test_captions = [
        "a cat on a table",
        "medieval armor",
    ]

    for test_caption in test_captions:
        generate_augmented_prompt(model, tokenizer, test_caption, device=device)


if __name__ == "__main__":
    main()
