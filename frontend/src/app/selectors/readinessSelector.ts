import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { RootState } from '../../app/store';
import { activeTabNameSelector } from '../../features/options/optionsSelectors';
import { OptionsState } from '../../features/options/optionsSlice';

import { SystemState } from '../../features/system/systemSlice';
import { InpaintingState } from '../../features/tabs/Inpainting/inpaintingSlice';
import { validateSeedWeights } from '../../common/util/seedWeightPairs';

export const readinessSelector = createSelector(
  [
    (state: RootState) => state.options,
    (state: RootState) => state.system,
    (state: RootState) => state.inpainting,
    activeTabNameSelector,
  ],
  (
    options: OptionsState,
    system: SystemState,
    inpainting: InpaintingState,
    activeTabName
  ) => {
    const {
      prompt,
      shouldGenerateVariations,
      seedWeights,
      maskPath,
      initialImage,
      seed,
    } = options;

    const { isProcessing, isConnected } = system;

    const { imageToInpaint } = inpainting;

    // Cannot generate without a prompt
    if (!prompt || Boolean(prompt.match(/^[\s\r\n]+$/))) {
      return false;
    }

    if (activeTabName === 'img2img' && !initialImage) {
      return false;
    }

    if (activeTabName === 'inpainting' && !imageToInpaint) {
      return false;
    }

    //  Cannot generate with a mask without img2img
    if (maskPath && !initialImage) {
      return false;
    }

    // TODO: job queue
    // Cannot generate if already processing an image
    if (isProcessing) {
      return false;
    }

    // Cannot generate if not connected
    if (!isConnected) {
      return false;
    }

    // Cannot generate variations without valid seed weights
    if (
      shouldGenerateVariations &&
      (!(validateSeedWeights(seedWeights) || seedWeights === '') || seed === -1)
    ) {
      return false;
    }

    // All good
    return true;
  },
  {
    memoizeOptions: {
      equalityCheck: _.isEqual,
      resultEqualityCheck: _.isEqual,
    },
  }
);
