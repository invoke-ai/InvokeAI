import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { RootState } from '../store';
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
      // maskPath,
      initialImage,
      seed,
    } = options;

    const { isProcessing, isConnected } = system;

    const { imageToInpaint } = inpainting;

    let isReady = true;
    const reasonsWhyNotReady: string[] = [];

    // Cannot generate without a prompt
    if (!prompt || Boolean(prompt.match(/^[\s\r\n]+$/))) {
      isReady = false;
      reasonsWhyNotReady.push('Missing prompt');
    }

    if (activeTabName === 'img2img' && !initialImage) {
      isReady = false;
      reasonsWhyNotReady.push('No initial image selected');
    }

    if (activeTabName === 'inpainting' && !imageToInpaint) {
      isReady = false;
      reasonsWhyNotReady.push('No inpainting image selected');
    }

    // // We don't use mask paths now.
    // //  Cannot generate with a mask without img2img
    // if (maskPath && !initialImage) {
    //   isReady = false;
    //   reasonsWhyNotReady.push(
    //     'On ImageToImage tab, but no mask is provided.'
    //   );
    // }

    // TODO: job queue
    // Cannot generate if already processing an image
    if (isProcessing) {
      isReady = false;
      reasonsWhyNotReady.push('System Busy');
    }

    // Cannot generate if not connected
    if (!isConnected) {
      isReady = false;
      reasonsWhyNotReady.push('System Disconnected');
    }

    // Cannot generate variations without valid seed weights
    if (
      shouldGenerateVariations &&
      (!(validateSeedWeights(seedWeights) || seedWeights === '') || seed === -1)
    ) {
      isReady = false;
      reasonsWhyNotReady.push('Seed-Weights badly formatted.');
    }

    // All good
    return { isReady, reasonsWhyNotReady };
  },
  {
    memoizeOptions: {
      equalityCheck: _.isEqual,
      resultEqualityCheck: _.isEqual,
    },
  }
);
