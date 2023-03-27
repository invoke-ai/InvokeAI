def standard_fashion_prompt(
    nationality="",
    garment_type="",
    material="",
    color="",
    style="",
    size="",
    closure="",
    lining="",
    collar="",
    sleeve_style="",
    hemline="",
    neckline="",
    waistline="",
    fit="",
    texture="",
    pattern="",
    brand="",
):
    """
    Inputs:
    nationality: string
    garment_type: string
    material: string
    color: string
    style: string
    size: string
    closure: string
    lining: string
    collar: string
    sleeve_style: string
    hemline: string
    neckline: string
    waistline: string
    fit: string
    texture: string
    pattern: string
    brand: string

    Returns:
    prompt: string
    negative_prompt: string
    """
    prompt = f"modelshoot style, RAW candid cinema, a {size} {nationality} woman wearing {color} {material} {style} {garment_type} with {pattern} design in {brand} fashion ad campaign, {fit}, {texture}, {closure}, {lining}, {collar}, {sleeve_style}, {hemline}, {neckline}, {waistline}, studio, 16mm, color graded portra 400 film, remarkable color, ultra realistic, textured skin, remarkable detailed pupils, realistic dull skin noise, visible skin detail, skin fuzz, dry skin, shot with cinematic camera, full body pose"

    negative_prompt = "B&W, logo, Glasses, Watermark, bad artist, blur, blurry, text, b&w, 3d, bad art, poorly drawn, disfigured, deformed, extra limbs, ugly hands, extra fingers, canvas frame, cartoon, 3d, disfigured, bad art, deformed, extra limbs, weird colors, blurry, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, out of frame, ugly, extra limbs, bad anatomy, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, fused fingers, too many fingers, long neck, Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d render"

    return prompt, negative_prompt
