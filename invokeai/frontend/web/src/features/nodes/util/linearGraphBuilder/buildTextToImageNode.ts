import { v4 as uuidv4 } from 'uuid';
import { RootState } from 'app/store';
import { TextToImageInvocation } from 'services/api';

export const buildTxt2ImgNode = (state: RootState): TextToImageInvocation => {
  const nodeId = uuidv4();
  const { generation, models } = state;

  const { selectedModelName } = models;

  const {
    prompt,
    negativePrompt,
    seed,
    steps,
    width,
    height,
    cfgScale: cfg_scale,
    sampler,
    seamless,
    shouldRandomizeSeed,
  } = generation;

  const textToImageNode: NonNullable<TextToImageInvocation> = {
    id: nodeId,
    type: 'txt2img',
    prompt: negativePrompt ? `${prompt} [${negativePrompt}]` : prompt,
    steps,
    width,
    height,
    cfg_scale,
    scheduler: sampler as TextToImageInvocation['scheduler'],
    seamless,
    model: selectedModelName,
    progress_images: true,
  };

  if (!shouldRandomizeSeed) {
    textToImageNode.seed = seed;
  }

  return textToImageNode;
};
