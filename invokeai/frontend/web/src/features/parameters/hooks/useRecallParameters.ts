import { createSelector } from '@reduxjs/toolkit';
import { useAppToaster } from 'app/components/Toaster';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import {
  controlAdapterRecalled,
  controlAdaptersReset,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import {
  ControlNetConfig,
  IPAdapterConfig,
  T2IAdapterConfig,
} from 'features/controlAdapters/store/types';
import {
  initialControlNet,
  initialIPAdapter,
  initialT2IAdapter,
} from 'features/controlAdapters/util/buildControlAdapter';
import {
  ControlNetMetadataItem,
  CoreMetadata,
  IPAdapterMetadataItem,
  LoRAMetadataItem,
  T2IAdapterMetadataItem,
} from 'features/nodes/types/types';
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
import { ImageDTO } from 'services/api/types';
import { v4 as uuidv4 } from 'uuid';
import {
  controlNetModelsAdapter,
  ipAdapterModelsAdapter,
  loraModelsAdapter,
  t2iAdapterModelsAdapter,
  useGetControlNetModelsQuery,
  useGetIPAdapterModelsQuery,
  useGetLoRAModelsQuery,
  useGetT2IAdapterModelsQuery,
} from '../../../services/api/endpoints/models';
import { loraRecalled, lorasCleared } from '../../lora/store/loraSlice';
import { initialImageSelected, modelSelected } from '../store/actions';
import {
  setCfgScale,
  setHeight,
  setHrfEnabled,
  setHrfMethod,
  setHrfStrength,
  setImg2imgStrength,
  setNegativePrompt,
  setPositivePrompt,
  setScheduler,
  setSeed,
  setSteps,
  setWidth,
  vaeSelected,
} from '../store/generationSlice';
import {
  isValidBoolean,
  isValidCfgScale,
  isValidControlNetModel,
  isValidHeight,
  isValidHrfMethod,
  isValidIPAdapterModel,
  isValidLoRAModel,
  isValidMainModel,
  isValidNegativePrompt,
  isValidPositivePrompt,
  isValidSDXLNegativeStylePrompt,
  isValidSDXLPositiveStylePrompt,
  isValidSDXLRefinerModel,
  isValidSDXLRefinerNegativeAestheticScore,
  isValidSDXLRefinerPositiveAestheticScore,
  isValidSDXLRefinerStart,
  isValidScheduler,
  isValidSeed,
  isValidSteps,
  isValidStrength,
  isValidVaeModel,
  isValidWidth,
} from '../types/parameterSchemas';

const selector = createSelector(
  stateSelector,
  ({ generation }) => generation.model,
  defaultSelectorOptions
);

export const useRecallParameters = () => {
  const dispatch = useAppDispatch();
  const toaster = useAppToaster();
  const { t } = useTranslation();
  const model = useAppSelector(selector);

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
    (
      positivePrompt: unknown,
      negativePrompt: unknown,
      positiveStylePrompt: unknown,
      negativeStylePrompt: unknown
    ) => {
      if (
        isValidPositivePrompt(positivePrompt) ||
        isValidNegativePrompt(negativePrompt) ||
        isValidSDXLPositiveStylePrompt(positiveStylePrompt) ||
        isValidSDXLNegativeStylePrompt(negativeStylePrompt)
      ) {
        if (isValidPositivePrompt(positivePrompt)) {
          dispatch(setPositivePrompt(positivePrompt));
        }

        if (isValidNegativePrompt(negativePrompt)) {
          dispatch(setNegativePrompt(negativePrompt));
        }

        if (isValidSDXLPositiveStylePrompt(positiveStylePrompt)) {
          dispatch(setPositiveStylePromptSDXL(positiveStylePrompt));
        }

        if (isValidSDXLPositiveStylePrompt(negativeStylePrompt)) {
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
   * Recall SDXL Positive Style Prompt with toast
   */
  const recallSDXLPositiveStylePrompt = useCallback(
    (positiveStylePrompt: unknown) => {
      if (!isValidSDXLPositiveStylePrompt(positiveStylePrompt)) {
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
      if (!isValidSDXLNegativeStylePrompt(negativeStylePrompt)) {
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
   * Recall vae model
   */
  const recallVaeModel = useCallback(
    (vae: unknown) => {
      if (!isValidVaeModel(vae) && !isNil(vae)) {
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

  /**
   * Recall high resolution enabled with toast
   */
  const recallHrfEnabled = useCallback(
    (hrfEnabled: unknown) => {
      if (!isValidBoolean(hrfEnabled)) {
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
      if (!isValidStrength(hrfStrength)) {
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
      if (!isValidHrfMethod(hrfMethod)) {
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
    (loraMetadataItem: LoRAMetadataItem) => {
      if (!isValidLoRAModel(loraMetadataItem.lora)) {
        return { lora: null, error: 'Invalid LoRA model' };
      }

      const { base_model, model_name } = loraMetadataItem.lora;

      const matchingLoRA = loraModels
        ? loraModelsAdapter
            .getSelectors()
            .selectById(loraModels, `${base_model}/lora/${model_name}`)
        : undefined;

      if (!matchingLoRA) {
        return { lora: null, error: 'LoRA model is not installed' };
      }

      const isCompatibleBaseModel =
        matchingLoRA?.base_model === model?.base_model;

      if (!isCompatibleBaseModel) {
        return {
          lora: null,
          error: 'LoRA incompatible with currently-selected model',
        };
      }

      return { lora: matchingLoRA, error: null };
    },
    [loraModels, model?.base_model]
  );

  const recallLoRA = useCallback(
    (loraMetadataItem: LoRAMetadataItem) => {
      const result = prepareLoRAMetadataItem(loraMetadataItem);

      if (!result.lora) {
        parameterNotSetToast(result.error);
        return;
      }

      dispatch(
        loraRecalled({ ...result.lora, weight: loraMetadataItem.weight })
      );

      parameterSetToast();
    },
    [prepareLoRAMetadataItem, dispatch, parameterSetToast, parameterNotSetToast]
  );

  /**
   * Recall ControlNet with toast
   */

  const { data: controlNetModels } = useGetControlNetModelsQuery(undefined);

  const prepareControlNetMetadataItem = useCallback(
    (controlnetMetadataItem: ControlNetMetadataItem) => {
      if (!isValidControlNetModel(controlnetMetadataItem.control_model)) {
        return { controlnet: null, error: 'Invalid ControlNet model' };
      }

      const {
        image,
        control_model,
        control_weight,
        begin_step_percent,
        end_step_percent,
        control_mode,
        resize_mode,
      } = controlnetMetadataItem;

      const matchingControlNetModel = controlNetModels
        ? controlNetModelsAdapter
            .getSelectors()
            .selectById(
              controlNetModels,
              `${control_model.base_model}/controlnet/${control_model.model_name}`
            )
        : undefined;

      if (!matchingControlNetModel) {
        return { controlnet: null, error: 'ControlNet model is not installed' };
      }

      const isCompatibleBaseModel =
        matchingControlNetModel?.base_model === model?.base_model;

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
        weight:
          typeof control_weight === 'number'
            ? control_weight
            : initialControlNet.weight,
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
    [controlNetModels, model?.base_model]
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
    [
      prepareControlNetMetadataItem,
      dispatch,
      parameterSetToast,
      parameterNotSetToast,
    ]
  );

  /**
   * Recall T2I Adapter with toast
   */

  const { data: t2iAdapterModels } = useGetT2IAdapterModelsQuery(undefined);

  const prepareT2IAdapterMetadataItem = useCallback(
    (t2iAdapterMetadataItem: T2IAdapterMetadataItem) => {
      if (!isValidControlNetModel(t2iAdapterMetadataItem.t2i_adapter_model)) {
        return { controlnet: null, error: 'Invalid ControlNet model' };
      }

      const {
        image,
        t2i_adapter_model,
        weight,
        begin_step_percent,
        end_step_percent,
        resize_mode,
      } = t2iAdapterMetadataItem;

      const matchingT2IAdapterModel = t2iAdapterModels
        ? t2iAdapterModelsAdapter
            .getSelectors()
            .selectById(
              t2iAdapterModels,
              `${t2i_adapter_model.base_model}/t2i_adapter/${t2i_adapter_model.model_name}`
            )
        : undefined;

      if (!matchingT2IAdapterModel) {
        return { controlnet: null, error: 'ControlNet model is not installed' };
      }

      const isCompatibleBaseModel =
        matchingT2IAdapterModel?.base_model === model?.base_model;

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
    [model?.base_model, t2iAdapterModels]
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
    [
      prepareT2IAdapterMetadataItem,
      dispatch,
      parameterSetToast,
      parameterNotSetToast,
    ]
  );

  /**
   * Recall IP Adapter with toast
   */

  const { data: ipAdapterModels } = useGetIPAdapterModelsQuery(undefined);

  const prepareIPAdapterMetadataItem = useCallback(
    (ipAdapterMetadataItem: IPAdapterMetadataItem) => {
      if (!isValidIPAdapterModel(ipAdapterMetadataItem?.ip_adapter_model)) {
        return { ipAdapter: null, error: 'Invalid IP Adapter model' };
      }

      const {
        image,
        ip_adapter_model,
        weight,
        begin_step_percent,
        end_step_percent,
      } = ipAdapterMetadataItem;

      const matchingIPAdapterModel = ipAdapterModels
        ? ipAdapterModelsAdapter
            .getSelectors()
            .selectById(
              ipAdapterModels,
              `${ip_adapter_model.base_model}/ip_adapter/${ip_adapter_model.model_name}`
            )
        : undefined;

      if (!matchingIPAdapterModel) {
        return { ipAdapter: null, error: 'IP Adapter model is not installed' };
      }

      const isCompatibleBaseModel =
        matchingIPAdapterModel?.base_model === model?.base_model;

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
    [ipAdapterModels, model?.base_model]
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
    [
      prepareIPAdapterMetadataItem,
      dispatch,
      parameterSetToast,
      parameterNotSetToast,
    ]
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
      if (isValidVaeModel(vae) || isNil(vae)) {
        if (isNil(vae)) {
          dispatch(vaeSelected(null));
        } else {
          dispatch(vaeSelected(vae));
        }
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

      if (isValidBoolean(hrf_enabled)) {
        dispatch(setHrfEnabled(hrf_enabled));
      }

      if (isValidStrength(hrf_strength)) {
        dispatch(setHrfStrength(hrf_strength));
      }

      if (isValidHrfMethod(hrf_method)) {
        dispatch(setHrfMethod(hrf_method));
      }

      if (isValidSDXLPositiveStylePrompt(positive_style_prompt)) {
        dispatch(setPositiveStylePromptSDXL(positive_style_prompt));
      }

      if (isValidSDXLNegativeStylePrompt(negative_style_prompt)) {
        dispatch(setNegativeStylePromptSDXL(negative_style_prompt));
      }

      if (isValidSDXLRefinerModel(refiner_model)) {
        dispatch(refinerModelChanged(refiner_model));
      }

      if (isValidSteps(refiner_steps)) {
        dispatch(setRefinerSteps(refiner_steps));
      }

      if (isValidCfgScale(refiner_cfg_scale)) {
        dispatch(setRefinerCFGScale(refiner_cfg_scale));
      }

      if (isValidScheduler(refiner_scheduler)) {
        dispatch(setRefinerScheduler(refiner_scheduler));
      }

      if (
        isValidSDXLRefinerPositiveAestheticScore(
          refiner_positive_aesthetic_score
        )
      ) {
        dispatch(
          setRefinerPositiveAestheticScore(refiner_positive_aesthetic_score)
        );
      }

      if (
        isValidSDXLRefinerNegativeAestheticScore(
          refiner_negative_aesthetic_score
        )
      ) {
        dispatch(
          setRefinerNegativeAestheticScore(refiner_negative_aesthetic_score)
        );
      }

      if (isValidSDXLRefinerStart(refiner_start)) {
        dispatch(setRefinerStart(refiner_start));
      }

      dispatch(lorasCleared());
      loras?.forEach((lora) => {
        const result = prepareLoRAMetadataItem(lora);
        if (result.lora) {
          dispatch(loraRecalled({ ...result.lora, weight: lora.weight }));
        }
      });

      dispatch(controlAdaptersReset());
      controlnets?.forEach((controlnet) => {
        const result = prepareControlNetMetadataItem(controlnet);
        if (result.controlnet) {
          dispatch(controlAdapterRecalled(result.controlnet));
        }
      });

      ipAdapters?.forEach((ipAdapter) => {
        const result = prepareIPAdapterMetadataItem(ipAdapter);
        if (result.ipAdapter) {
          dispatch(controlAdapterRecalled(result.ipAdapter));
        }
      });

      t2iAdapters?.forEach((t2iAdapter) => {
        const result = prepareT2IAdapterMetadataItem(t2iAdapter);
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
    recallModel,
    recallScheduler,
    recallVaeModel,
    recallSteps,
    recallWidth,
    recallHeight,
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
