import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { useMemo } from 'react';
import { useAppSelector } from '../../app/store';
import { RootState } from '../../app/store';
import { activeTabNameSelector } from '../../features/options/optionsSelectors';
import { OptionsState } from '../../features/options/optionsSlice';

import { SystemState } from '../../features/system/systemSlice';
import { InpaintingState } from '../../features/tabs/Inpainting/inpaintingSlice';
import { validateSeedWeights } from '../util/seedWeightPairs';

export const useCheckParametersSelector = createSelector(
  [
    (state: RootState) => state.options,
    (state: RootState) => state.system,
    (state: RootState) => state.inpainting,
    activeTabNameSelector
  ],
  (options: OptionsState, system: SystemState, inpainting: InpaintingState, activeTabName) => {
    return {
      // options
      prompt: options.prompt,
      shouldGenerateVariations: options.shouldGenerateVariations,
      seedWeights: options.seedWeights,
      maskPath: options.maskPath,
      initialImage: options.initialImage,
      seed: options.seed,
      activeTabName,
      // system
      isProcessing: system.isProcessing,
      isConnected: system.isConnected,
      // inpainting
      hasInpaintingImage: Boolean(inpainting.imageToInpaint),
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);
/**
 * Checks relevant pieces of state to confirm generation will not deterministically fail.
 * This is used to prevent the 'Generate' button from being clicked.
 */
const useCheckParameters = (): boolean => {
  const {
    prompt,
    shouldGenerateVariations,
    seedWeights,
    maskPath,
    initialImage,
    seed,
    activeTabName,
    isProcessing,
    isConnected,
    hasInpaintingImage,
  } = useAppSelector(useCheckParametersSelector);

  return useMemo(() => {
    // Cannot generate without a prompt
    if (!prompt || Boolean(prompt.match(/^[\s\r\n]+$/))) {
      return false;
    }

    if (activeTabName === 'img2img' && !initialImage) {
      return false;
    }

    if (activeTabName === 'inpainting' && !hasInpaintingImage) {
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
  }, [
    prompt,
    maskPath,
    isProcessing,
    initialImage,
    isConnected,
    shouldGenerateVariations,
    seedWeights,
    seed,
    activeTabName,
    hasInpaintingImage,
  ]);
};

export default useCheckParameters;
