import { v4 as uuidv4 } from 'uuid';
import { RootState } from 'app/store';
import { ImageToImageInvocation } from 'services/api';
import { _Image } from 'app/invokeai';
import { initialImageSelector } from 'features/parameters/store/generationSelectors';

export const buildImg2ImgNode = (
  state: RootState
): Record<string, ImageToImageInvocation> => {
  const nodeId = uuidv4();
  const { generation, system, models } = state;

  const { shouldDisplayInProgressType } = system;
  const { currentModel: model } = models;

  const {
    prompt,
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
    throw 'no initial image';
  }

  return {
    [nodeId]: {
      id: nodeId,
      type: 'img2img',
      prompt,
      seed: shouldRandomizeSeed ? -1 : seed,
      steps,
      width,
      height,
      cfg_scale: cfgScale,
      sampler_name: sampler as ImageToImageInvocation['sampler_name'],
      seamless,
      model,
      progress_images: shouldDisplayInProgressType === 'full-res',
      image: {
        image_name: initialImage.name,
        image_type: initialImage.type,
      },
      strength,
      fit,
    },
  };
};
