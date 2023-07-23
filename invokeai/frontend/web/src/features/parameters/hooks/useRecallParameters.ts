import { useAppToaster } from 'app/components/Toaster';
import { useAppDispatch } from 'app/store/storeHooks';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { UnsafeImageMetadata } from 'services/api/endpoints/images';
import { ImageDTO } from 'services/api/types';
import { initialImageSelected, modelSelected } from '../store/actions';
import {
  setCfgScale,
  setHeight,
  setImg2imgStrength,
  setNegativePrompt,
  setPositivePrompt,
  setScheduler,
  setSeed,
  setSteps,
  setWidth,
} from '../store/generationSlice';
import {
  isValidCfgScale,
  isValidHeight,
  isValidMainModel,
  isValidNegativePrompt,
  isValidPositivePrompt,
  isValidScheduler,
  isValidSeed,
  isValidSteps,
  isValidStrength,
  isValidWidth,
} from '../types/parameterSchemas';

export const useRecallParameters = () => {
  const dispatch = useAppDispatch();
  const toaster = useAppToaster();
  const { t } = useTranslation();

  const parameterSetToast = useCallback(() => {
    toaster({
      title: t('toast.parameterSet'),
      status: 'info',
      duration: 2500,
      isClosable: true,
    });
  }, [t, toaster]);

  const parameterNotSetToast = useCallback(() => {
    toaster({
      title: t('toast.parameterNotSet'),
      status: 'warning',
      duration: 2500,
      isClosable: true,
    });
  }, [t, toaster]);

  const allParameterSetToast = useCallback(() => {
    toaster({
      title: t('toast.parametersSet'),
      status: 'info',
      duration: 2500,
      isClosable: true,
    });
  }, [t, toaster]);

  const allParameterNotSetToast = useCallback(() => {
    toaster({
      title: t('toast.parametersNotSet'),
      status: 'warning',
      duration: 2500,
      isClosable: true,
    });
  }, [t, toaster]);

  /**
   * Recall both prompts with toast
   */
  const recallBothPrompts = useCallback(
    (positivePrompt: unknown, negativePrompt: unknown) => {
      if (
        isValidPositivePrompt(positivePrompt) ||
        isValidNegativePrompt(negativePrompt)
      ) {
        if (isValidPositivePrompt(positivePrompt)) {
          dispatch(setPositivePrompt(positivePrompt));
        }
        if (isValidNegativePrompt(negativePrompt)) {
          dispatch(setNegativePrompt(negativePrompt));
        }
        parameterSetToast();
        return;
      }
      parameterNotSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  /**
   * Recall positive prompt with toast
   */
  const recallPositivePrompt = useCallback(
    (positivePrompt: unknown) => {
      if (!isValidPositivePrompt(positivePrompt)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setPositivePrompt(positivePrompt));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  /**
   * Recall negative prompt with toast
   */
  const recallNegativePrompt = useCallback(
    (negativePrompt: unknown) => {
      if (!isValidNegativePrompt(negativePrompt)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setNegativePrompt(negativePrompt));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  /**
   * Recall seed with toast
   */
  const recallSeed = useCallback(
    (seed: unknown) => {
      if (!isValidSeed(seed)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setSeed(seed));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  /**
   * Recall CFG scale with toast
   */
  const recallCfgScale = useCallback(
    (cfgScale: unknown) => {
      if (!isValidCfgScale(cfgScale)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setCfgScale(cfgScale));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  /**
   * Recall model with toast
   */
  const recallModel = useCallback(
    (model: unknown) => {
      if (!isValidMainModel(model)) {
        parameterNotSetToast();
        return;
      }
      dispatch(modelSelected(model));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  /**
   * Recall scheduler with toast
   */
  const recallScheduler = useCallback(
    (scheduler: unknown) => {
      if (!isValidScheduler(scheduler)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setScheduler(scheduler));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  /**
   * Recall steps with toast
   */
  const recallSteps = useCallback(
    (steps: unknown) => {
      if (!isValidSteps(steps)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setSteps(steps));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  /**
   * Recall width with toast
   */
  const recallWidth = useCallback(
    (width: unknown) => {
      if (!isValidWidth(width)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setWidth(width));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  /**
   * Recall height with toast
   */
  const recallHeight = useCallback(
    (height: unknown) => {
      if (!isValidHeight(height)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setHeight(height));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  /**
   * Recall strength with toast
   */
  const recallStrength = useCallback(
    (strength: unknown) => {
      if (!isValidStrength(strength)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setImg2imgStrength(strength));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  /*
   * Sets image as initial image with toast
   */
  const sendToImageToImage = useCallback(
    (image: ImageDTO) => {
      dispatch(initialImageSelected(image));
    },
    [dispatch]
  );

  const recallAllParameters = useCallback(
    (metadata: UnsafeImageMetadata['metadata'] | undefined) => {
      if (!metadata) {
        allParameterNotSetToast();
        return;
      }

      const {
        cfg_scale,
        height,
        model,
        positive_prompt,
        negative_prompt,
        scheduler,
        seed,
        steps,
        width,
        strength,
      } = metadata;

      if (isValidCfgScale(cfg_scale)) {
        dispatch(setCfgScale(cfg_scale));
      }
      if (isValidMainModel(model)) {
        dispatch(modelSelected(model));
      }
      if (isValidPositivePrompt(positive_prompt)) {
        dispatch(setPositivePrompt(positive_prompt));
      }
      if (isValidNegativePrompt(negative_prompt)) {
        dispatch(setNegativePrompt(negative_prompt));
      }
      if (isValidScheduler(scheduler)) {
        dispatch(setScheduler(scheduler));
      }
      if (isValidSeed(seed)) {
        dispatch(setSeed(seed));
      }
      if (isValidSteps(steps)) {
        dispatch(setSteps(steps));
      }
      if (isValidWidth(width)) {
        dispatch(setWidth(width));
      }
      if (isValidHeight(height)) {
        dispatch(setHeight(height));
      }
      if (isValidStrength(strength)) {
        dispatch(setImg2imgStrength(strength));
      }

      allParameterSetToast();
    },
    [allParameterNotSetToast, allParameterSetToast, dispatch]
  );

  return {
    recallBothPrompts,
    recallPositivePrompt,
    recallNegativePrompt,
    recallSeed,
    recallCfgScale,
    recallModel,
    recallScheduler,
    recallSteps,
    recallWidth,
    recallHeight,
    recallStrength,
    recallAllParameters,
    sendToImageToImage,
  };
};
