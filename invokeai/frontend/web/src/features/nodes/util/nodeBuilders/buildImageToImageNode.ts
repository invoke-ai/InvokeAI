import { v4 as uuidv4 } from 'uuid';
import { RootState } from 'app/store/store';
import {
  Edge,
  ImageToImageInvocation,
  TextToImageInvocation,
} from 'services/api';
import { O } from 'ts-toolbelt';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';

export const buildImg2ImgNode = (
  state: RootState,
  overrides: O.Partial<ImageToImageInvocation, 'deep'> = {}
): ImageToImageInvocation => {
  const nodeId = uuidv4();
  const { generation } = state;

  const activeTabName = activeTabNameSelector(state);

  const {
    positivePrompt: prompt,
    negativePrompt: negativePrompt,
    seed,
    steps,
    width,
    height,
    cfgScale,
    scheduler,
    model,
    img2imgStrength: strength,
    shouldFitToWidthHeight: fit,
    shouldRandomizeSeed,
    initialImage,
  } = generation;

  // const initialImage = initialImageSelector(state);

  const imageToImageNode: ImageToImageInvocation = {
    id: nodeId,
    type: 'img2img',
    prompt: `${prompt} [${negativePrompt}]`,
    steps,
    width,
    height,
    cfg_scale: cfgScale,
    scheduler,
    model,
    strength,
    fit,
  };

  // on Canvas tab, we do not manually specific init image
  if (activeTabName !== 'unifiedCanvas') {
    if (!initialImage) {
      // TODO: handle this more better
      throw 'no initial image';
    }

    imageToImageNode.image = {
      image_name: initialImage.name,
      image_origin: initialImage.type,
    };
  }

  if (!shouldRandomizeSeed) {
    imageToImageNode.seed = seed;
  }

  Object.assign(imageToImageNode, overrides);

  return imageToImageNode;
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
        fit: true,
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
