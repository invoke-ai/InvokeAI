import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import {
  FrontendToBackendParametersConfig,
  frontendToBackendParameters,
} from 'common/util/parameterTranslation';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { postprocessingSelector } from 'features/parameters/store/postprocessingSelectors';
import { systemSelector } from 'features/system/store/systemSelectors';
import { canvasSelector } from '../store/canvasSelectors';
import { useCallback, useMemo } from 'react';

const selector = createSelector(
  [generationSelector, postprocessingSelector, systemSelector, canvasSelector],
  (generation, postprocessing, system, canvas) => {
    const frontendToBackendParametersConfig: FrontendToBackendParametersConfig =
      {
        generationMode: 'unifiedCanvas',
        generationState: generation,
        postprocessingState: postprocessing,
        canvasState: canvas,
        systemState: system,
      };

    return frontendToBackendParametersConfig;
  }
);

export const usePrepareCanvasState = () => {
  const frontendToBackendParametersConfig = useAppSelector(selector);

  const getGenerationParameters = useCallback(() => {
    const { generationParameters, esrganParameters, facetoolParameters } =
      frontendToBackendParameters(frontendToBackendParametersConfig);
    console.log(generationParameters);
  }, [frontendToBackendParametersConfig]);

  return getGenerationParameters;
};
