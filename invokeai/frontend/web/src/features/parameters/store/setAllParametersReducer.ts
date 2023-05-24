import { Draft, PayloadAction } from '@reduxjs/toolkit';
import { GenerationState } from './generationSlice';
import { ImageDTO, ImageToImageInvocation } from 'services/api';
import { isScheduler } from 'app/constants';

export const setAllParametersReducer = (
  state: Draft<GenerationState>,
  action: PayloadAction<ImageDTO | undefined>
) => {
  const node = action.payload?.metadata.invokeai?.node;

  if (!node) {
    return;
  }

  if (
    node.type === 'txt2img' ||
    node.type === 'img2img' ||
    node.type === 'inpaint'
  ) {
    const { cfg_scale, height, model, prompt, scheduler, seed, steps, width } =
      node;

    if (cfg_scale !== undefined) {
      state.cfgScale = Number(cfg_scale);
    }
    if (height !== undefined) {
      state.height = Number(height);
    }
    if (model !== undefined) {
      state.model = String(model);
    }
    if (prompt !== undefined) {
      state.positivePrompt = String(prompt);
    }
    if (scheduler !== undefined) {
      const schedulerString = String(scheduler);
      if (isScheduler(schedulerString)) {
        state.scheduler = schedulerString;
      }
    }
    if (seed !== undefined) {
      state.seed = Number(seed);
      state.shouldRandomizeSeed = false;
    }
    if (steps !== undefined) {
      state.steps = Number(steps);
    }
    if (width !== undefined) {
      state.width = Number(width);
    }
  }

  if (node.type === 'img2img') {
    const { fit, image } = node as ImageToImageInvocation;

    if (fit !== undefined) {
      state.shouldFitToWidthHeight = Boolean(fit);
    }
    // if (image !== undefined) {
    //   state.initialImage = image;
    // }
  }
};
