import { useAppToaster } from 'app/components/Toaster';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { controlAdapterRecalled, controlAdaptersReset } from 'features/controlAdapters/store/controlAdaptersSlice';
import { setHrfEnabled, setHrfMethod, setHrfStrength } from 'features/hrf/store/hrfSlice';
import { loraRecalled, lorasCleared } from 'features/lora/store/loraSlice';
import { isModelIdentifier } from 'features/nodes/types/common';
import type {
  ControlNetMetadataItem,
  CoreMetadata,
  IPAdapterMetadataItem,
  LoRAMetadataItem,
  T2IAdapterMetadataItem,
} from 'features/nodes/types/metadata';
import { initialImageSelected, modelSelected } from 'features/parameters/store/actions';
import {
  heightRecalled,
  selectGenerationSlice,
  setCfgRescaleMultiplier,
  setCfgScale,
  setImg2imgStrength,
  setNegativePrompt,
  setPositivePrompt,
  setScheduler,
  setSeed,
  setSteps,
  vaeSelected,
  widthRecalled,
} from 'features/parameters/store/generationSlice';
import type { ParameterModel } from 'features/parameters/types/parameterSchemas';
import {
  isParameterCFGRescaleMultiplier,
  isParameterCFGScale,
  isParameterHeight,
  isParameterHRFEnabled,
  isParameterHRFMethod,
  isParameterNegativePrompt,
  isParameterNegativeStylePromptSDXL,
  isParameterPositivePrompt,
  isParameterPositiveStylePromptSDXL,
  isParameterScheduler,
  isParameterSDXLRefinerModel,
  isParameterSDXLRefinerNegativeAestheticScore,
  isParameterSDXLRefinerPositiveAestheticScore,
  isParameterSDXLRefinerStart,
  isParameterSeed,
  isParameterSteps,
  isParameterStrength,
  isParameterWidth,
} from 'features/parameters/types/parameterSchemas';
import {
  prepareControlNetMetadataItem,
  prepareIPAdapterMetadataItem,
  prepareLoRAMetadataItem,
  prepareMainModelMetadataItem,
  prepareT2IAdapterMetadataItem,
  prepareVAEMetadataItem,
} from 'features/parameters/util/modelMetadataHelpers';
import {
  refinerModelChanged,
  setNegativeStylePromptSDXL,
  setPositiveStylePromptSDXL,
  setRefinerCFGScale,
  setRefinerNegativeAestheticScore,
  setRefinerPositiveAestheticScore,
  setRefinerScheduler,
  setRefinerStart,
  setRefinerSteps,
} from 'features/sdxl/store/sdxlSlice';
import { isNil } from 'lodash-es';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import type { ImageDTO } from 'services/api/types';

const selectModel = createMemoizedSelector(selectGenerationSlice, (generation) => generation.model);

export const useRecallParameters = () => {
  const dispatch = useAppDispatch();
  const toaster = useAppToaster();
  const { t } = useTranslation();
  const model = useAppSelector(selectModel);

  const parameterSetToast = useCallback(() => {
    toaster({
      title: t('toast.parameterSet'),
      status: 'info',
      duration: 2500,
      isClosable: true,
    });
  }, [t, toaster]);

  const parameterNotSetToast = useCallback(
    (description?: string) => {
      toaster({
        title: t('toast.parameterNotSet'),
        description,
        status: 'warning',
        duration: 2500,
        isClosable: true,
      });
    },
    [t, toaster]
  );

  const allParameterSetToast = useCallback(() => {
    toaster({
      title: t('toast.parametersSet'),
      status: 'info',
      duration: 2500,
      isClosable: true,
    });
  }, [t, toaster]);

  const allParameterNotSetToast = useCallback(
    (description?: string) => {
      toaster({
        title: t('toast.parametersNotSet'),
        status: 'warning',
        description,
        duration: 2500,
        isClosable: true,
      });
    },
    [t, toaster]
  );

  const recallBothPrompts = useCallback(
    (positivePrompt: unknown, negativePrompt: unknown, positiveStylePrompt: unknown, negativeStylePrompt: unknown) => {
      if (
        isParameterPositivePrompt(positivePrompt) ||
        isParameterNegativePrompt(negativePrompt) ||
        isParameterPositiveStylePromptSDXL(positiveStylePrompt) ||
        isParameterNegativeStylePromptSDXL(negativeStylePrompt)
      ) {
        if (isParameterPositivePrompt(positivePrompt)) {
          dispatch(setPositivePrompt(positivePrompt));
        }

        if (isParameterNegativePrompt(negativePrompt)) {
          dispatch(setNegativePrompt(negativePrompt));
        }

        if (isParameterPositiveStylePromptSDXL(positiveStylePrompt)) {
          dispatch(setPositiveStylePromptSDXL(positiveStylePrompt));
        }

        if (isParameterPositiveStylePromptSDXL(negativeStylePrompt)) {
          dispatch(setNegativeStylePromptSDXL(negativeStylePrompt));
        }

        parameterSetToast();
        return;
      }
      parameterNotSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  const recallPositivePrompt = useCallback(
    (positivePrompt: unknown) => {
      if (!isParameterPositivePrompt(positivePrompt)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setPositivePrompt(positivePrompt));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  const recallNegativePrompt = useCallback(
    (negativePrompt: unknown) => {
      if (!isParameterNegativePrompt(negativePrompt)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setNegativePrompt(negativePrompt));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  const recallSDXLPositiveStylePrompt = useCallback(
    (positiveStylePrompt: unknown) => {
      if (!isParameterPositiveStylePromptSDXL(positiveStylePrompt)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setPositiveStylePromptSDXL(positiveStylePrompt));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  const recallSDXLNegativeStylePrompt = useCallback(
    (negativeStylePrompt: unknown) => {
      if (!isParameterNegativeStylePromptSDXL(negativeStylePrompt)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setNegativeStylePromptSDXL(negativeStylePrompt));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  const recallSeed = useCallback(
    (seed: unknown) => {
      if (!isParameterSeed(seed)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setSeed(seed));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  const recallCfgScale = useCallback(
    (cfgScale: unknown) => {
      if (!isParameterCFGScale(cfgScale)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setCfgScale(cfgScale));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  const recallCfgRescaleMultiplier = useCallback(
    (cfgRescaleMultiplier: unknown) => {
      if (!isParameterCFGRescaleMultiplier(cfgRescaleMultiplier)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setCfgRescaleMultiplier(cfgRescaleMultiplier));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  const recallScheduler = useCallback(
    (scheduler: unknown) => {
      if (!isParameterScheduler(scheduler)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setScheduler(scheduler));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  const recallSteps = useCallback(
    (steps: unknown) => {
      if (!isParameterSteps(steps)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setSteps(steps));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  const recallWidth = useCallback(
    (width: unknown) => {
      if (!isParameterWidth(width)) {
        parameterNotSetToast();
        return;
      }
      dispatch(widthRecalled(width));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  const recallHeight = useCallback(
    (height: unknown) => {
      if (!isParameterHeight(height)) {
        parameterNotSetToast();
        return;
      }
      dispatch(heightRecalled(height));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  const recallWidthAndHeight = useCallback(
    (width: unknown, height: unknown) => {
      if (!isParameterWidth(width)) {
        allParameterNotSetToast();
        return;
      }
      if (!isParameterHeight(height)) {
        allParameterNotSetToast();
        return;
      }
      dispatch(heightRecalled(height));
      dispatch(widthRecalled(width));
      allParameterSetToast();
    },
    [dispatch, allParameterSetToast, allParameterNotSetToast]
  );

  const recallStrength = useCallback(
    (strength: unknown) => {
      if (!isParameterStrength(strength)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setImg2imgStrength(strength));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  const recallHrfEnabled = useCallback(
    (hrfEnabled: unknown) => {
      if (!isParameterHRFEnabled(hrfEnabled)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setHrfEnabled(hrfEnabled));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  const recallHrfStrength = useCallback(
    (hrfStrength: unknown) => {
      if (!isParameterStrength(hrfStrength)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setHrfStrength(hrfStrength));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  const recallHrfMethod = useCallback(
    (hrfMethod: unknown) => {
      if (!isParameterHRFMethod(hrfMethod)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setHrfMethod(hrfMethod));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  const recallModel = useCallback(
    async (modelMetadataItem: unknown) => {
      try {
        const model = await prepareMainModelMetadataItem(modelMetadataItem);
        dispatch(modelSelected(model));
        parameterSetToast();
      } catch (e) {
        parameterNotSetToast((e as unknown as Error).message);
        return;
      }
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  const recallVaeModel = useCallback(
    async (vaeMetadataItem: unknown) => {
      if (isNil(vaeMetadataItem)) {
        dispatch(vaeSelected(null));
        parameterSetToast();
        return;
      }
      try {
        const vae = await prepareVAEMetadataItem(vaeMetadataItem);
        dispatch(vaeSelected(vae));
        parameterSetToast();
      } catch (e) {
        parameterNotSetToast((e as unknown as Error).message);
        return;
      }
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  const recallLoRA = useCallback(
    async (loraMetadataItem: LoRAMetadataItem) => {
      try {
        const lora = await prepareLoRAMetadataItem(loraMetadataItem, model?.base);
        dispatch(loraRecalled(lora));
        parameterSetToast();
      } catch (e) {
        parameterNotSetToast((e as unknown as Error).message);
        return;
      }
    },
    [model?.base, dispatch, parameterSetToast, parameterNotSetToast]
  );

  const recallControlNet = useCallback(
    async (controlnetMetadataItem: ControlNetMetadataItem) => {
      try {
        const controlNetConfig = await prepareControlNetMetadataItem(controlnetMetadataItem, model?.base);
        dispatch(controlAdapterRecalled(controlNetConfig));
        parameterSetToast();
      } catch (e) {
        parameterNotSetToast((e as unknown as Error).message);
        return;
      }
    },
    [model?.base, dispatch, parameterSetToast, parameterNotSetToast]
  );

  const recallT2IAdapter = useCallback(
    async (t2iAdapterMetadataItem: T2IAdapterMetadataItem) => {
      try {
        const t2iAdapterConfig = await prepareT2IAdapterMetadataItem(t2iAdapterMetadataItem, model?.base);
        dispatch(controlAdapterRecalled(t2iAdapterConfig));
        parameterSetToast();
      } catch (e) {
        parameterNotSetToast((e as unknown as Error).message);
        return;
      }
    },
    [model?.base, dispatch, parameterSetToast, parameterNotSetToast]
  );

  const recallIPAdapter = useCallback(
    async (ipAdapterMetadataItem: IPAdapterMetadataItem) => {
      try {
        const ipAdapterConfig = await prepareIPAdapterMetadataItem(ipAdapterMetadataItem, model?.base);
        dispatch(controlAdapterRecalled(ipAdapterConfig));
        parameterSetToast();
      } catch (e) {
        parameterNotSetToast((e as unknown as Error).message);
        return;
      }
    },
    [model?.base, dispatch, parameterSetToast, parameterNotSetToast]
  );

  const sendToImageToImage = useCallback(
    (image: ImageDTO) => {
      dispatch(initialImageSelected(image));
    },
    [dispatch]
  );

  const recallAllParameters = useCallback(
    async (metadata: CoreMetadata | undefined) => {
      if (!metadata) {
        allParameterNotSetToast();
        return;
      }

      const {
        cfg_scale,
        cfg_rescale_multiplier,
        height,
        model,
        positive_prompt,
        negative_prompt,
        scheduler,
        vae,
        seed,
        steps,
        width,
        strength,
        hrf_enabled,
        hrf_strength,
        hrf_method,
        positive_style_prompt,
        negative_style_prompt,
        refiner_model,
        refiner_cfg_scale,
        refiner_steps,
        refiner_scheduler,
        refiner_positive_aesthetic_score,
        refiner_negative_aesthetic_score,
        refiner_start,
        loras,
        controlnets,
        ipAdapters,
        t2iAdapters,
      } = metadata;

      let newModel: ParameterModel | undefined = undefined;

      if (isModelIdentifier(model)) {
        try {
          const _model = await prepareMainModelMetadataItem(model);
          dispatch(modelSelected(_model));
          newModel = _model;
        } catch {
          return;
        }
      }

      if (isParameterCFGScale(cfg_scale)) {
        dispatch(setCfgScale(cfg_scale));
      }

      if (isParameterCFGRescaleMultiplier(cfg_rescale_multiplier)) {
        dispatch(setCfgRescaleMultiplier(cfg_rescale_multiplier));
      }

      if (isParameterPositivePrompt(positive_prompt)) {
        dispatch(setPositivePrompt(positive_prompt));
      }

      if (isParameterNegativePrompt(negative_prompt)) {
        dispatch(setNegativePrompt(negative_prompt));
      }

      if (isParameterScheduler(scheduler)) {
        dispatch(setScheduler(scheduler));
      }
      if (isModelIdentifier(vae) || isNil(vae)) {
        if (isNil(vae)) {
          dispatch(vaeSelected(null));
        } else {
          try {
            const _vae = await prepareVAEMetadataItem(vae, newModel?.base);
            dispatch(vaeSelected(_vae));
          } catch {
            return;
          }
        }
      }

      if (isParameterSeed(seed)) {
        dispatch(setSeed(seed));
      }

      if (isParameterSteps(steps)) {
        dispatch(setSteps(steps));
      }

      if (isParameterWidth(width)) {
        dispatch(widthRecalled(width));
      }

      if (isParameterHeight(height)) {
        dispatch(heightRecalled(height));
      }

      if (isParameterStrength(strength)) {
        dispatch(setImg2imgStrength(strength));
      }

      if (isParameterHRFEnabled(hrf_enabled)) {
        dispatch(setHrfEnabled(hrf_enabled));
      }

      if (isParameterStrength(hrf_strength)) {
        dispatch(setHrfStrength(hrf_strength));
      }

      if (isParameterHRFMethod(hrf_method)) {
        dispatch(setHrfMethod(hrf_method));
      }

      if (isParameterPositiveStylePromptSDXL(positive_style_prompt)) {
        dispatch(setPositiveStylePromptSDXL(positive_style_prompt));
      }

      if (isParameterNegativeStylePromptSDXL(negative_style_prompt)) {
        dispatch(setNegativeStylePromptSDXL(negative_style_prompt));
      }

      if (isParameterSDXLRefinerModel(refiner_model)) {
        dispatch(refinerModelChanged(refiner_model));
      }

      if (isParameterSteps(refiner_steps)) {
        dispatch(setRefinerSteps(refiner_steps));
      }

      if (isParameterCFGScale(refiner_cfg_scale)) {
        dispatch(setRefinerCFGScale(refiner_cfg_scale));
      }

      if (isParameterScheduler(refiner_scheduler)) {
        dispatch(setRefinerScheduler(refiner_scheduler));
      }

      if (isParameterSDXLRefinerPositiveAestheticScore(refiner_positive_aesthetic_score)) {
        dispatch(setRefinerPositiveAestheticScore(refiner_positive_aesthetic_score));
      }

      if (isParameterSDXLRefinerNegativeAestheticScore(refiner_negative_aesthetic_score)) {
        dispatch(setRefinerNegativeAestheticScore(refiner_negative_aesthetic_score));
      }

      if (isParameterSDXLRefinerStart(refiner_start)) {
        dispatch(setRefinerStart(refiner_start));
      }

      dispatch(lorasCleared());
      loras?.forEach(async (loraMetadataItem) => {
        try {
          const lora = await prepareLoRAMetadataItem(loraMetadataItem, newModel?.base);
          dispatch(loraRecalled(lora));
        } catch {
          return;
        }
      });

      dispatch(controlAdaptersReset());
      controlnets?.forEach(async (controlNetMetadataItem) => {
        try {
          const controlNet = await prepareControlNetMetadataItem(controlNetMetadataItem, newModel?.base);
          dispatch(controlAdapterRecalled(controlNet));
        } catch {
          return;
        }
      });

      ipAdapters?.forEach(async (ipAdapterMetadataItem) => {
        try {
          const ipAdapter = await prepareIPAdapterMetadataItem(ipAdapterMetadataItem, newModel?.base);
          dispatch(controlAdapterRecalled(ipAdapter));
        } catch {
          return;
        }
      });

      t2iAdapters?.forEach(async (t2iAdapterMetadataItem) => {
        try {
          const t2iAdapter = await prepareT2IAdapterMetadataItem(t2iAdapterMetadataItem, newModel?.base);
          dispatch(controlAdapterRecalled(t2iAdapter));
        } catch {
          return;
        }
      });

      allParameterSetToast();
    },
    [dispatch, allParameterSetToast, allParameterNotSetToast]
  );

  return {
    recallBothPrompts,
    recallPositivePrompt,
    recallNegativePrompt,
    recallSDXLPositiveStylePrompt,
    recallSDXLNegativeStylePrompt,
    recallSeed,
    recallCfgScale,
    recallCfgRescaleMultiplier,
    recallModel,
    recallScheduler,
    recallVaeModel,
    recallSteps,
    recallWidth,
    recallHeight,
    recallWidthAndHeight,
    recallStrength,
    recallHrfEnabled,
    recallHrfStrength,
    recallHrfMethod,
    recallLoRA,
    recallControlNet,
    recallIPAdapter,
    recallT2IAdapter,
    recallAllParameters,
    sendToImageToImage,
  };
};
