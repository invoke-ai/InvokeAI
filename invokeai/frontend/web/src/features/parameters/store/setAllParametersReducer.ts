import { Draft, PayloadAction } from '@reduxjs/toolkit';
import { Image } from 'app/types/invokeai';
import { GenerationState } from './generationSlice';
import { ImageToImageInvocation } from 'services/api';

export const setAllParametersReducer = (
  state: Draft<GenerationState>,
  action: PayloadAction<Image | undefined>
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
      state.prompt = String(prompt);
    }
    if (scheduler !== undefined) {
      state.sampler = String(scheduler);
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
