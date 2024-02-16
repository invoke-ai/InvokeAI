import { useAppToaster } from 'app/components/Toaster';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import { controlAdapterRecalled, controlAdaptersReset } from 'features/controlAdapters/store/controlAdaptersSlice';
import type { ControlNetConfig, IPAdapterConfig, T2IAdapterConfig } from 'features/controlAdapters/store/types';
import {
  initialControlNet,
  initialIPAdapter,
  initialT2IAdapter,
} from 'features/controlAdapters/util/buildControlAdapter';
import { setHrfEnabled, setHrfMethod, setHrfStrength } from 'features/hrf/store/hrfSlice';
import { loraRecalled, lorasCleared } from 'features/lora/store/loraSlice';
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
  isParameterControlNetModel,
  isParameterHeight,
  isParameterHRFEnabled,
  isParameterHRFMethod,
  isParameterIPAdapterModel,
  isParameterLoRAModel,
  isParameterModel,
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
  isParameterVAEModel,
  isParameterWidth,
} from 'features/parameters/types/parameterSchemas';
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
import {
  controlNetModelsAdapterSelectors,
  ipAdapterModelsAdapterSelectors,
  loraModelsAdapterSelectors,
  t2iAdapterModelsAdapterSelectors,
  useGetControlNetModelsQuery,
  useGetIPAdapterModelsQuery,
  useGetLoRAModelsQuery,
  useGetT2IAdapterModelsQuery,
} from 'services/api/endpoints/models';
import type { ImageDTO } from 'services/api/types';
import { v4 as uuidv4 } from 'uuid';

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

  /**
   * Recall both prompts with toast
   */
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

  /**
   * Recall positive prompt with toast
   */
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

  /**
   * Recall negative prompt with toast
   */
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

  /**
   * Recall SDXL Positive Style Prompt with toast
   */
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

  /**
   * Recall SDXL Negative Style Prompt with toast
   */
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

  /**
   * Recall seed with toast
   */
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

  /**
   * Recall CFG scale with toast
   */
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

  /**
   * Recall CFG rescale multiplier with toast
   */
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

  /**
   * Recall model with toast
   */
  const recallModel = useCallback(
    (model: unknown) => {
      if (!isParameterModel(model)) {
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
      if (!isParameterScheduler(scheduler)) {
        parameterNotSetToast();
        return;
      }
      dispatch(setScheduler(scheduler));
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  /**
   * Recall vae model
   */
  const recallVaeModel = useCallback(
    (vae: unknown) => {
      if (!isParameterVAEModel(vae) && !isNil(vae)) {
        parameterNotSetToast();
        return;
      }
      if (isNil(vae)) {
        dispatch(vaeSelected(null));
      } else {
        dispatch(vaeSelected(vae));
      }
      parameterSetToast();
    },
    [dispatch, parameterSetToast, parameterNotSetToast]
  );

  /**
   * Recall steps with toast
   */
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

  /**
   * Recall width with toast
   */
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

  /**
   * Recall height with toast
   */
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

  /**
   * Recall width and height with toast
   */
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

  /**
   * Recall strength with toast
   */
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

  /**
   * Recall high resolution enabled with toast
   */
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

  /**
   * Recall high resolution strength with toast
   */
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

  /**
   * Recall high resolution method with toast
   */
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

  /**
   * Recall LoRA with toast
   */

  const { data: loraModels } = useGetLoRAModelsQuery(undefined);

  const prepareLoRAMetadataItem = useCallback(
    (loraMetadataItem: LoRAMetadataItem, newModel?: ParameterModel) => {
      if (!isParameterLoRAModel(loraMetadataItem.lora)) {
        return { lora: null, error: 'Invalid LoRA model' };
      }

      const { lora } = loraMetadataItem;

      const matchingLoRA = loraModels ? loraModelsAdapterSelectors.selectById(loraModels, lora.key) : undefined;

      if (!matchingLoRA) {
        return { lora: null, error: 'LoRA model is not installed' };
      }

      const isCompatibleBaseModel = matchingLoRA?.base === (newModel ?? model)?.base;

      if (!isCompatibleBaseModel) {
        return {
          lora: null,
          error: 'LoRA incompatible with currently-selected model',
        };
      }

      return { lora: matchingLoRA, error: null };
    },
    [loraModels, model]
  );

  const recallLoRA = useCallback(
    (loraMetadataItem: LoRAMetadataItem) => {
      const result = prepareLoRAMetadataItem(loraMetadataItem);

      if (!result.lora) {
        parameterNotSetToast(result.error);
        return;
      }

      dispatch(loraRecalled({ ...result.lora, weight: loraMetadataItem.weight }));

      parameterSetToast();
    },
    [prepareLoRAMetadataItem, dispatch, parameterSetToast, parameterNotSetToast]
  );

  /**
   * Recall ControlNet with toast
   */

  const { data: controlNetModels } = useGetControlNetModelsQuery(undefined);

  const prepareControlNetMetadataItem = useCallback(
    (controlnetMetadataItem: ControlNetMetadataItem, newModel?: ParameterModel) => {
      if (!isParameterControlNetModel(controlnetMetadataItem.control_model)) {
        return { controlnet: null, error: 'Invalid ControlNet model' };
      }

      const { image, control_model, control_weight, begin_step_percent, end_step_percent, control_mode, resize_mode } =
        controlnetMetadataItem;

      const matchingControlNetModel = controlNetModels
        ? controlNetModelsAdapterSelectors.selectById(controlNetModels, control_model.key)
        : undefined;

      if (!matchingControlNetModel) {
        return { controlnet: null, error: 'ControlNet model is not installed' };
      }

      const isCompatibleBaseModel = matchingControlNetModel?.base === (newModel ?? model)?.base;

      if (!isCompatibleBaseModel) {
        return {
          controlnet: null,
          error: 'ControlNet incompatible with currently-selected model',
        };
      }

      // We don't save the original image that was processed into a control image, only the processed image
      const processorType = 'none';
      const processorNode = CONTROLNET_PROCESSORS.none.default;

      const controlnet: ControlNetConfig = {
        type: 'controlnet',
        isEnabled: true,
        model: matchingControlNetModel,
        weight: typeof control_weight === 'number' ? control_weight : initialControlNet.weight,
        beginStepPct: begin_step_percent || initialControlNet.beginStepPct,
        endStepPct: end_step_percent || initialControlNet.endStepPct,
        controlMode: control_mode || initialControlNet.controlMode,
        resizeMode: resize_mode || initialControlNet.resizeMode,
        controlImage: image?.image_name || null,
        processedControlImage: image?.image_name || null,
        processorType,
        processorNode,
        shouldAutoConfig: true,
        id: uuidv4(),
      };

      return { controlnet, error: null };
    },
    [controlNetModels, model]
  );

  const recallControlNet = useCallback(
    (controlnetMetadataItem: ControlNetMetadataItem) => {
      const result = prepareControlNetMetadataItem(controlnetMetadataItem);

      if (!result.controlnet) {
        parameterNotSetToast(result.error);
        return;
      }

      dispatch(controlAdapterRecalled(result.controlnet));

      parameterSetToast();
    },
    [prepareControlNetMetadataItem, dispatch, parameterSetToast, parameterNotSetToast]
  );

  /**
   * Recall T2I Adapter with toast
   */

  const { data: t2iAdapterModels } = useGetT2IAdapterModelsQuery(undefined);

  const prepareT2IAdapterMetadataItem = useCallback(
    (t2iAdapterMetadataItem: T2IAdapterMetadataItem, newModel?: ParameterModel) => {
      if (!isParameterControlNetModel(t2iAdapterMetadataItem.t2i_adapter_model)) {
        return { controlnet: null, error: 'Invalid ControlNet model' };
      }

      const { image, t2i_adapter_model, weight, begin_step_percent, end_step_percent, resize_mode } =
        t2iAdapterMetadataItem;

      const matchingT2IAdapterModel = t2iAdapterModels
        ? t2iAdapterModelsAdapterSelectors.selectById(t2iAdapterModels, t2i_adapter_model.key)
        : undefined;

      if (!matchingT2IAdapterModel) {
        return { controlnet: null, error: 'ControlNet model is not installed' };
      }

      const isCompatibleBaseModel = matchingT2IAdapterModel?.base === (newModel ?? model)?.base;

      if (!isCompatibleBaseModel) {
        return {
          t2iAdapter: null,
          error: 'ControlNet incompatible with currently-selected model',
        };
      }

      // We don't save the original image that was processed into a control image, only the processed image
      const processorType = 'none';
      const processorNode = CONTROLNET_PROCESSORS.none.default;

      const t2iAdapter: T2IAdapterConfig = {
        type: 't2i_adapter',
        isEnabled: true,
        model: matchingT2IAdapterModel,
        weight: typeof weight === 'number' ? weight : initialT2IAdapter.weight,
        beginStepPct: begin_step_percent || initialT2IAdapter.beginStepPct,
        endStepPct: end_step_percent || initialT2IAdapter.endStepPct,
        resizeMode: resize_mode || initialT2IAdapter.resizeMode,
        controlImage: image?.image_name || null,
        processedControlImage: image?.image_name || null,
        processorType,
        processorNode,
        shouldAutoConfig: true,
        id: uuidv4(),
      };

      return { t2iAdapter, error: null };
    },
    [model, t2iAdapterModels]
  );

  const recallT2IAdapter = useCallback(
    (t2iAdapterMetadataItem: T2IAdapterMetadataItem) => {
      const result = prepareT2IAdapterMetadataItem(t2iAdapterMetadataItem);

      if (!result.t2iAdapter) {
        parameterNotSetToast(result.error);
        return;
      }

      dispatch(controlAdapterRecalled(result.t2iAdapter));

      parameterSetToast();
    },
    [prepareT2IAdapterMetadataItem, dispatch, parameterSetToast, parameterNotSetToast]
  );

  /**
   * Recall IP Adapter with toast
   */

  const { data: ipAdapterModels } = useGetIPAdapterModelsQuery(undefined);

  const prepareIPAdapterMetadataItem = useCallback(
    (ipAdapterMetadataItem: IPAdapterMetadataItem, newModel?: ParameterModel) => {
      if (!isParameterIPAdapterModel(ipAdapterMetadataItem?.ip_adapter_model)) {
        return { ipAdapter: null, error: 'Invalid IP Adapter model' };
      }

      const { image, ip_adapter_model, weight, begin_step_percent, end_step_percent } = ipAdapterMetadataItem;

      const matchingIPAdapterModel = ipAdapterModels
        ? ipAdapterModelsAdapterSelectors.selectById(ipAdapterModels, ip_adapter_model.key)
        : undefined;

      if (!matchingIPAdapterModel) {
        return { ipAdapter: null, error: 'IP Adapter model is not installed' };
      }

      const isCompatibleBaseModel = matchingIPAdapterModel?.base === (newModel ?? model)?.base;

      if (!isCompatibleBaseModel) {
        return {
          ipAdapter: null,
          error: 'IP Adapter incompatible with currently-selected model',
        };
      }

      const ipAdapter: IPAdapterConfig = {
        id: uuidv4(),
        type: 'ip_adapter',
        isEnabled: true,
        controlImage: image?.image_name ?? null,
        model: matchingIPAdapterModel,
        weight: weight ?? initialIPAdapter.weight,
        beginStepPct: begin_step_percent ?? initialIPAdapter.beginStepPct,
        endStepPct: end_step_percent ?? initialIPAdapter.endStepPct,
      };

      return { ipAdapter, error: null };
    },
    [ipAdapterModels, model]
  );

  const recallIPAdapter = useCallback(
    (ipAdapterMetadataItem: IPAdapterMetadataItem) => {
      const result = prepareIPAdapterMetadataItem(ipAdapterMetadataItem);

      if (!result.ipAdapter) {
        parameterNotSetToast(result.error);
        return;
      }

      dispatch(controlAdapterRecalled(result.ipAdapter));

      parameterSetToast();
    },
    [prepareIPAdapterMetadataItem, dispatch, parameterSetToast, parameterNotSetToast]
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
    (metadata: CoreMetadata | undefined) => {
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

      if (isParameterModel(model)) {
        newModel = model;
        dispatch(modelSelected(model));
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
      if (isParameterVAEModel(vae) || isNil(vae)) {
        if (isNil(vae)) {
          dispatch(vaeSelected(null));
        } else {
          dispatch(vaeSelected(vae));
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
      loras?.forEach((lora) => {
        const result = prepareLoRAMetadataItem(lora, newModel);
        if (result.lora) {
          dispatch(loraRecalled({ ...result.lora, weight: lora.weight }));
        }
      });

      dispatch(controlAdaptersReset());
      controlnets?.forEach((controlnet) => {
        const result = prepareControlNetMetadataItem(controlnet, newModel);
        if (result.controlnet) {
          dispatch(controlAdapterRecalled(result.controlnet));
        }
      });

      ipAdapters?.forEach((ipAdapter) => {
        const result = prepareIPAdapterMetadataItem(ipAdapter, newModel);
        if (result.ipAdapter) {
          dispatch(controlAdapterRecalled(result.ipAdapter));
        }
      });

      t2iAdapters?.forEach((t2iAdapter) => {
        const result = prepareT2IAdapterMetadataItem(t2iAdapter, newModel);
        if (result.t2iAdapter) {
          dispatch(controlAdapterRecalled(result.t2iAdapter));
        }
      });

      allParameterSetToast();
    },
    [
      dispatch,
      allParameterSetToast,
      allParameterNotSetToast,
      prepareLoRAMetadataItem,
      prepareControlNetMetadataItem,
      prepareIPAdapterMetadataItem,
      prepareT2IAdapterMetadataItem,
    ]
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
