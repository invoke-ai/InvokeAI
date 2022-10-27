import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { useMemo } from 'react';
import { useAppSelector } from '../../app/store';
import { RootState } from '../../app/store';
import { GalleryState } from '../../features/gallery/gallerySlice';
import { OptionsState } from '../../features/options/optionsSlice';

import { SystemState } from '../../features/system/systemSlice';
import { InpaintingState } from '../../features/tabs/Inpainting/inpaintingSlice';
import { tabMap } from '../../features/tabs/InvokeTabs';
import { validateSeedWeights } from '../util/seedWeightPairs';

export const optionsSelector = createSelector(
  (state: RootState) => state.options,
  (options: OptionsState) => {
    return {
      prompt: options.prompt,
      shouldGenerateVariations: options.shouldGenerateVariations,
      seedWeights: options.seedWeights,
      maskPath: options.maskPath,
      initialImagePath: options.initialImagePath,
      seed: options.seed,
      activeTabName: tabMap[options.activeTab],
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export const systemSelector = createSelector(
  (state: RootState) => state.system,
  (system: SystemState) => {
    return {
      isProcessing: system.isProcessing,
      isConnected: system.isConnected,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export const inpaintingSelector = createSelector(
  (state: RootState) => state.inpainting,
  (inpainting: InpaintingState) => {
    return {
      isMaskEmpty: inpainting.lines.length === 0,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export const gallerySelector = createSelector(
  (state: RootState) => state.gallery,
  (gallery: GalleryState) => {
    return {
      hasCurrentImage: Boolean(gallery.currentImage),
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
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
    initialImagePath,
    seed,
    activeTabName,
  } = useAppSelector(optionsSelector);

  const { isProcessing, isConnected } = useAppSelector(systemSelector);

  const { isMaskEmpty } = useAppSelector(inpaintingSelector);

  const { hasCurrentImage } = useAppSelector(gallerySelector);

  return useMemo(() => {
    // Cannot generate without a prompt
    if (!prompt || Boolean(prompt.match(/^[\s\r\n]+$/))) {
      return false;
    }

    if (activeTabName === 'img2img' && !initialImagePath) {
      return false;
    }

    if (activeTabName === 'inpainting' && (!hasCurrentImage || isMaskEmpty)) {
      return false;
    }

    //  Cannot generate with a mask without img2img
    if (maskPath && !initialImagePath) {
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
    initialImagePath,
    isProcessing,
    isConnected,
    shouldGenerateVariations,
    seedWeights,
    seed,
    activeTabName,
    hasCurrentImage,
    isMaskEmpty,
  ]);
};

export default useCheckParameters;
