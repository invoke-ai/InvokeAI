import { v4 as uuidv4 } from 'uuid';
import { RootState } from 'app/store';
import { TextToImageInvocation } from 'services/api';
import { _Image } from 'app/invokeai';

export function buildTxt2ImgNode(
  state: RootState
): Record<string, TextToImageInvocation> {
  const nodeId = uuidv4();
  const { generation, system } = state;

  const { shouldDisplayInProgressType, model } = system;

  const {
    prompt,
    seed,
    steps,
    width,
    height,
    cfgScale: cfg_scale,
    sampler,
    seamless,
    shouldRandomizeSeed,
  } = generation;

  // missing fields in TextToImageInvocation: strength, hires_fix
  return {
    [nodeId]: {
      id: nodeId,
      type: 'txt2img',
      prompt,
      seed: shouldRandomizeSeed ? -1 : seed,
      steps,
      width,
      height,
      cfg_scale,
      sampler_name: sampler as TextToImageInvocation['sampler_name'],
      seamless,
      model,
      progress_images: shouldDisplayInProgressType === 'full-res',
    },
  };
}
