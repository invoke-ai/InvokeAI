export type SDXLStylePreset =
  | 'enhance'
  | 'anime_style'
  | 'photo_realistic'
  | 'digital_art'
  | 'comic_book'
  | 'fantasy_art'
  | 'analog_film'
  | 'neon_punk'
  | 'isometric'
  | 'low_poly'
  | 'origami'
  | 'line_art'
  | 'craft_clay'
  | 'cinematic'
  | '3d_model'
  | 'pixel_art'
  | 'texture';

type SDXLStylePresetFormat = {
  positive: string;
  negative: string;
};

export const sdxlStylePresets: Record<SDXLStylePreset, SDXLStylePresetFormat> =
  {
    enhance: {
      positive:
        'breathtaking {positive_prompt}, award-winning, professional, highly detailed',
      negative:
        '{negative_prompt}, ugly, deformed, noisy, blurry, distorted, grainy',
    },
    anime_style: {
      positive:
        'anime artwork {positive_prompt}, anime style, key visual, vibrant, studio anime, highly detailed',
      negative:
        '{negative_prompt}, photo, deformed, black and white, realism, disfigured, low contrast',
    },
    photo_realistic: {
      positive:
        'cinematic photo {positive_prompt}, 35mm photograph, film, bokeh, professional, 4k, highly detailed',
      negative:
        '{negative_prompt}, drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly',
    },
    digital_art: {
      positive:
        'concept art {positive_prompt}, digital artwork, illustrative, painterly, matte painting, highly detailed',
      negative: '{negative_prompt}, photo, photorealistic, realism, ugly',
    },
    comic_book: {
      positive:
        'comic {positive_prompt}, graphic illustration, comic art, graphic novel art, vibrant, highly detailed',
      negative:
        '{negative_prompt}, photograph, deformed, glitch, noisy, realistic, stock photo',
    },
    fantasy_art: {
      positive:
        'ethereal fantasy concept art of {positive_prompt}, magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy',
      negative:
        '{negative_prompt}, photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white',
    },
    analog_film: {
      positive:
        'analog film photo {positive_prompt}, faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage',
      negative:
        '{negative_prompt}, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured',
    },
    neon_punk: {
      positive:
        'neonpunk style {positive_prompt}, cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional',
      negative:
        '{negative_prompt}, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured',
    },
    isometric: {
      positive:
        'isometric style {positive_prompt}, vibrant, beautiful, crisp, detailed, ultra detailed, intricate',
      negative:
        '{negative_prompt}, deformed, mutated, ugly, disfigured, blur, blurry, noise, noisy, realistic, photographic',
    },
    low_poly: {
      positive:
        'low-poly style {positive_prompt}, low-poly game art, polygon mesh, jagged, blocky, wireframe edges, centered composition',
      negative:
        '{negative_prompt}, noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo',
    },
    origami: {
      positive:
        'origami style {positive_prompt}, paper art, pleated paper, folded, origami art, pleats, cut and fold, centered composition',
      negative:
        '{negative_prompt}, noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo',
    },
    line_art: {
      positive:
        'line art drawing {positive_prompt}, professional, sleek, modern, minimalist, graphic, line art, vector graphics',
      negative:
        '{negative_prompt}, anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic',
    },
    craft_clay: {
      positive:
        'play-doh style {positive_prompt}, sculpture, clay art, centered composition, Claymation',
      negative:
        '{negative_prompt}, sloppy, messy, grainy, highly detailed, ultra textured, photo',
    },
    cinematic: {
      positive:
        'cinematic film still {positive_prompt}, shallow depth of field, vignette, highly detailed, high budget Hollywood movie, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy',
      negative:
        '{negative_prompt}, anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured',
    },
    '3d_model': {
      positive:
        'professional 3d model {positive_prompt}, octane render, highly detailed, volumetric, dramatic lighting',
      negative:
        '{negative_prompt}, ugly, deformed, noisy, low poly, blurry, painting',
    },
    pixel_art: {
      positive:
        'pixel-art {positive_prompt}, low-res, blocky, pixel art style, 8-bit graphics',
      negative:
        '{negative_prompt}, sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic',
    },
    texture: {
      positive: 'texture, {positive_prompt}, top down close-up',
      negative: '{negative_prompt}, ugly, deformed, noisy, blurry',
    },
  };
