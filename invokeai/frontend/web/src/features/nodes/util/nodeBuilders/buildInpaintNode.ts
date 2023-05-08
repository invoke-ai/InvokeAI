import { v4 as uuidv4 } from 'uuid';
import { RootState } from 'app/store/store';
import {
  Edge,
  ImageToImageInvocation,
  InpaintInvocation,
  TextToImageInvocation,
} from 'services/api';
import { initialImageSelector } from 'features/parameters/store/generationSelectors';
import { O } from 'ts-toolbelt';

export const buildInpaintNode = (
  state: RootState,
  overrides: O.Partial<InpaintInvocation, 'deep'> = {}
): InpaintInvocation => {
  const nodeId = uuidv4();
  const { generation, system, models } = state;

  const { selectedModelName } = models;

  const {
    prompt,
    negativePrompt,
    seed,
    steps,
    width,
    height,
    cfgScale,
    sampler,
    seamless,
    img2imgStrength: strength,
    shouldFitToWidthHeight: fit,
    shouldRandomizeSeed,
  } = generation;

  const initialImage = initialImageSelector(state);

  if (!initialImage) {
    // TODO: handle this
    // throw 'no initial image';
  }

  const imageToImageNode: InpaintInvocation = {
    id: nodeId,
    type: 'inpaint',
    prompt: `${prompt} [${negativePrompt}]`,
    steps,
    width,
    height,
    cfg_scale: cfgScale,
    scheduler: sampler as InpaintInvocation['scheduler'],
    seamless,
    model: selectedModelName,
    progress_images: true,
    image: initialImage
      ? {
          image_name: initialImage.name,
          image_type: initialImage.type,
        }
      : undefined,
    strength,
    fit,
  };

  if (!shouldRandomizeSeed) {
    imageToImageNode.seed = seed;
  }

  Object.assign(imageToImageNode, overrides);

  return imageToImageNode;
};
