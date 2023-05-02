import { v4 as uuidv4 } from 'uuid';
import { RootState } from 'app/store/store';
import { TextToImageInvocation } from 'services/api';
import { O } from 'ts-toolbelt';

export const buildTxt2ImgNode = (
  state: RootState,
  overrides: O.Partial<TextToImageInvocation, 'deep'> = {}
): TextToImageInvocation => {
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
    prompt: `${prompt} [${negativePrompt}]`,
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

  Object.assign(textToImageNode, overrides);

  return textToImageNode;
};
