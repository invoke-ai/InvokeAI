import { v4 as uuidv4 } from 'uuid';
import { RootState } from 'app/store';
import {
  Edge,
  ImageToImageInvocation,
  TextToImageInvocation,
} from 'services/api';
import { _Image } from 'app/invokeai';
import { initialImageSelector } from 'features/parameters/store/generationSelectors';

export const buildImg2ImgNode = (
  state: RootState
): Record<string, ImageToImageInvocation> => {
  const nodeId = uuidv4();
  const { generation, system, models } = state;

  const { shouldDisplayInProgressType } = system;
  const { selectedModelName } = models;

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

  const imageToImageNode: ImageToImageInvocation = {
    id: nodeId,
    type: 'img2img',
    prompt,
    seed: shouldRandomizeSeed ? -1 : seed,
    steps,
    width,
    height,
    cfg_scale: cfgScale,
    scheduler: sampler as ImageToImageInvocation['scheduler'],
    seamless,
    model: selectedModelName,
    progress_images: shouldDisplayInProgressType === 'full-res',
    image: {
      image_name: initialImage.name,
      image_type: initialImage.type,
    },
    strength,
    fit,
  };

  return {
    [nodeId]: imageToImageNode,
  };
};

type hiresReturnType = {
  node: Record<string, ImageToImageInvocation>;
  edge: Edge;
};

export const buildHiResNode = (
  baseNode: Record<string, TextToImageInvocation>,
  strength?: number
): hiresReturnType => {
  const nodeId = uuidv4();
  const baseNodeId = Object.keys(baseNode)[0];
  const baseNodeValues = Object.values(baseNode)[0];

  return {
    node: {
      [nodeId]: {
        ...baseNodeValues,
        id: nodeId,
        type: 'img2img',
        strength,
      },
    },
    edge: {
      source: {
        field: 'image',
        node_id: baseNodeId,
      },
      destination: {
        field: 'image',
        node_id: nodeId,
      },
    },
  };
};
