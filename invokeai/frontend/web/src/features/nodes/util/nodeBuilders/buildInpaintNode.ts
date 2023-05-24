import { v4 as uuidv4 } from 'uuid';
import { RootState } from 'app/store/store';
import { InpaintInvocation } from 'services/api';
import { O } from 'ts-toolbelt';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';

export const buildInpaintNode = (
  state: RootState,
  overrides: O.Partial<InpaintInvocation, 'deep'> = {}
): InpaintInvocation => {
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

  const inpaintNode: InpaintInvocation = {
    id: nodeId,
    type: 'inpaint',
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

    inpaintNode.image = {
      image_name: initialImage.name,
      image_type: initialImage.type,
    };
  }

  if (!shouldRandomizeSeed) {
    inpaintNode.seed = seed;
  }

  Object.assign(inpaintNode, overrides);

  return inpaintNode;
};
