import { Draft, PayloadAction } from '@reduxjs/toolkit';
import { GenerationState } from './generationSlice';
import { ImageDTO, ImageToImageInvocation } from 'services/api';
import { isScheduler } from 'app/constants';

export const setAllParametersReducer = (
  state: Draft<GenerationState>,
  action: PayloadAction<ImageDTO | undefined>
) => {
  const metadata = action.payload?.metadata;

  if (!metadata) {
    return;
  }

  // not sure what this list should be
  if (
    metadata.type === 't2l' ||
    metadata.type === 'l2l' ||
    metadata.type === 'inpaint'
  ) {
    const {
      cfg_scale,
      height,
      model,
      positive_conditioning,
      negative_conditioning,
      scheduler,
      seed,
      steps,
      width,
    } = metadata;

    if (cfg_scale !== undefined) {
      state.cfgScale = Number(cfg_scale);
    }
    if (height !== undefined) {
      state.height = Number(height);
    }
    if (model !== undefined) {
      state.model = String(model);
    }
    if (positive_conditioning !== undefined) {
      state.positivePrompt = String(positive_conditioning);
    }
    if (negative_conditioning !== undefined) {
      state.negativePrompt = String(negative_conditioning);
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

  if (metadata.type === 'l2l') {
    const { fit, image } = metadata as ImageToImageInvocation;

    if (fit !== undefined) {
      state.shouldFitToWidthHeight = Boolean(fit);
    }
    // if (image !== undefined) {
    //   state.initialImage = image;
    // }
  }
};
