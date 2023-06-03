import { v4 as uuidv4 } from 'uuid';
import { RootState } from 'app/store/store';
import { InpaintInvocation } from 'services/api';
import { O } from 'ts-toolbelt';

export const buildInpaintNode = (
  state: RootState,
  overrides: O.Partial<InpaintInvocation, 'deep'> = {}
): InpaintInvocation => {
  const nodeId = uuidv4();

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
  } = state.generation;

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

  if (!shouldRandomizeSeed) {
    inpaintNode.seed = seed;
  }

  Object.assign(inpaintNode, overrides);

  return inpaintNode;
};
