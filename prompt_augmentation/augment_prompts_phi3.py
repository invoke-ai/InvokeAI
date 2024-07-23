import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_augmented_prompt(model, tokenizer, initial_prompt, device: torch.device):
    instruction = f"""Your task is to translate a short image caption to a more detailed caption for the same image. The detailed caption should adhere to the following:
    - be 1 sentence long
    - use descriptive language and unique or distinct words that relate to the subject of interest
    - it may add new details, but shouldn't change the subject of the original caption

    Here are some examples:
    Original caption: "A cat on a table"
    Detailed caption: "A fluffy cat with a curious expression, sitting on a wooden table next to a vase of flowers."

    Original caption: "medieval armor"
    Detailed caption: "The gleaming suit of medieval armor stands proudly in the museum, its intricate engravings telling tales of long-forgotten battles and chivalry."

    Original caption: "A panda bear as a mad scientist"
    Detailed caption: "Clad in a tiny lab coat and goggles, the panda bear feverishly mixes colorful potions, embodying the eccentricity of a mad scientist in its whimsical laboratory."

    Original caption: "{initial_prompt}"
    Detailed caption:"""
    messages = [
        {
            "role": "user",
            "content": instruction,
        }
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = inputs.to(device)

    outputs = model.generate(inputs, max_new_tokens=200)
    text = tokenizer.batch_decode(outputs)[0]
    assert isinstance(text, str)

    output_prompt = text.split("<|assistant|>")[-1].strip()

    print(f"Original Prompt: '{initial_prompt}'\n\n")
    print(f"Generated text: '{output_prompt}'\n\n")
    print("----------------------")


def main():
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.float16)
    model = model.to(device=device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

    test_prompts = [
        # Simple
        "medieval armor",
        "make a calendar",
        "a hairless mouse with human ears",
        "A panda bear as a mad scientist",
        "Male portrait photo",
        "Apocalyptic scenes of a meteor storm over a volcano."
        "cinematic still of a stainless steel robot swimming in a pool",
        "Fantasy castle on a hilltop, sunset",
        "A bee devouring the world",
        "transparent ghost of a soldier in a cemetry",
        "Space dog",
        "A mermaid playing chess with a dolphin",
        "F18 hornet",
        "Dogs playing poker",
        "a strong and muscular warrior with a bow",
        "a masterpiece painting of a crocodile wearing a hoodie sitting on the roof of a car",
        "cozy Danish interior design with wooden floor modern realistic archviz scandinavian",
        "A curious cat exploring a haunted mansion",
        "toilet design toilet in style of dodge charger toilet, black, photo",
        "swirling water tornados epic fantasy",
        "Painting of melted gemstones metallic sculpture with electrifying god rays brane bejeweled style",
        "olympic swimming pool",
        # Detailed subject, simple style
        "a man looking like a lobster, with lobster hands, smiling, looking straight into the camera with arms wide spread",
        "A mythical monstrous black furry nocturnal dog with bear claws, green glistening scaled wings and glowing crimson eyes. Several heavy chains hang from its body and ankles.",
        "Photograph of a red apple on a wooden table while in the background a window is observed through which a flash of light enters"
        # Simple subject, detailed style
        "photo of a bicycle, detailed, 8k uhd, dslr, high quality, film grain, Fujifilm XT3",
        # Detailed subject and style
        "RAW photo, aristocratic russian noblewoman, dressed in medieval dress, model face, come hither gesture, medieval mansion, medieval nobility, slim body, high detailed skin, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
        "A close-up photograph of a fat orange cat with lasagna in its mouth. Shot on Leica M6.",
        "iOS app icon, cute white and yellow tiger head, happy, colorful, minimal, detailed, fantasy, app icon, 8k, white background",
        "Cinematic, off-center, two-shot, 35mm film still of a 30-year-old french man, curly brown hair and a stained beige polo sweater, reading a book to his adorable 5-year-old daughter, wearing fuzzy pink pajamas, sitting in a cozy corner nook, sunny natural lighting, sun shining through the glass of the window, warm morning glow, sharp focus, heavenly illumination, unconditional love",
        "futuristic robot but wearing medieval lord clothes and in a medieval castle, extremely detailed digital art, ambient lightning, interior, castle, medieval, painting, digital painting, trending on devianart, photorealistic, sunrays",
        "very dark focused flash photo, amazing quality, masterpiece, best quality, hyper detailed, ultra detailed, UHD, perfect anatomy, portrait, dof, hyper-realism, majestic, awesome, inspiring,Capture the thrilling showdown between the ancient mummy and the colossal sand boss in an epic battle amidst swirling dust and desert sands. Embrace the action and chaos as these formidable forces clash in the heart of the dunes. cinematic composition, soft shadows, national geographic style",
        "8k breathtaking view of a Boat on the waterside, Swiss alp, mountain lake, milky haze, morning lake mist, masterpiece, award-winning, professional, highly detailed in yvonne coomber style, undefined, snow, mist, in the distance, glowing eyes",
    ]

    for test_prompt in test_prompts:
        generate_augmented_prompt(model, tokenizer, test_prompt, device=device)


if __name__ == "__main__":
    main()
