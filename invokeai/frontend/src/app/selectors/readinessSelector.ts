import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { RootState } from 'app/store';
import { activeTabNameSelector } from 'features/options/store/optionsSelectors';
import { OptionsState } from 'features/options/store/optionsSlice';
import { SystemState } from 'features/system/store/systemSlice';
import { validateSeedWeights } from 'common/util/seedWeightPairs';
import { initialCanvasImageSelector } from 'features/canvas/store/canvasSelectors';

export const readinessSelector = createSelector(
  [
    (state: RootState) => state.options,
    (state: RootState) => state.system,
    initialCanvasImageSelector,
    activeTabNameSelector,
  ],
  (
    options: OptionsState,
    system: SystemState,
    initialCanvasImage,
    activeTabName
  ) => {
    const {
      prompt,
      shouldGenerateVariations,
      seedWeights,
      initialImage,
      seed,
    } = options;

    const { isProcessing, isConnected } = system;

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
