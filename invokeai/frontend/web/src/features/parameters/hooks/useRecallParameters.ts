import { createSelector } from '@reduxjs/toolkit';
import { useAppToaster } from 'app/components/Toaster';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  CoreMetadata,
  LoRAMetadataItem,
  ControlNetMetadataItem,
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
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { ImageDTO } from 'services/api/types';
import {
  controlNetModelsAdapter,
  loraModelsAdapter,
  useGetControlNetModelsQuery,
  useGetLoRAModelsQuery,
} from '../../../services/api/endpoints/models';
import {
  ControlNetConfig,
  controlNetEnabled,
  controlNetRecalled,
  controlNetReset,
  initialControlNet,
} from '../../controlNet/store/controlNetSlice';
import { loraRecalled, lorasCleared } from '../../lora/store/loraSlice';
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
  isValidLoRAModel,
  isValidControlNetModel,
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
  isValidWidth,
} from '../types/parameterSchemas';
import { v4 as uuidv4 } from 'uuid';
import {
  CONTROLNET_PROCESSORS,
  CONTROLNET_MODEL_DEFAULT_PROCESSORS,
} from 'features/controlNet/store/constants';
import { ControlNetProcessorType } from 'features/controlNet/store/types';

const selector = createSelector(stateSelector, ({ generation }) => {
  const { model } = generation;
  return { model };
});

export const useRecallParameters = () => {
  const dispatch = useAppDispatch();
  const toaster = useAppToaster();
  const { t } = useTranslation();
  const { model } = useAppSelector(selector);

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
   * Recall LoRA with toast
   */

  const { loras } = useGetLoRAModelsQuery(undefined, {
    selectFromResult: (result) => ({
      loras: result.data
        ? loraModelsAdapter.getSelectors().selectAll(result.data)
        : [],
    }),
  });

  const prepareLoRAMetadataItem = useCallback(
    (loraMetadataItem: LoRAMetadataItem) => {
      if (!isValidLoRAModel(loraMetadataItem.lora)) {
        return { lora: null, error: 'Invalid LoRA model' };
      }

      const { base_model, model_name } = loraMetadataItem.lora;

      const matchingLoRA = loras.find(
        (l) => l.base_model === base_model && l.model_name === model_name
      );

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
    [loras, model?.base_model]
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

  const { controlnets } = useGetControlNetModelsQuery(undefined, {
    selectFromResult: (result) => ({
      controlnets: result.data
        ? controlNetModelsAdapter.getSelectors().selectAll(result.data)
        : [],
    }),
  });

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

      const matchingControlNetModel = controlnets.find(
        (c) =>
          c.base_model === control_model.base_model &&
          c.model_name === control_model.model_name
      );

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

      const controlNetId = uuidv4();
      const processorType: ControlNetProcessorType =
        CONTROLNET_MODEL_DEFAULT_PROCESSORS[
          matchingControlNetModel.model_name
        ] || initialControlNet.processorType;
      const processorNode = CONTROLNET_PROCESSORS[processorType].default;

      const controlnet: ControlNetConfig = {
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
        processorNode:
          processorNode.type !== 'none'
            ? processorNode
            : initialControlNet.processorNode,
        shouldAutoConfig: true,
        controlNetId,
      };

      return { controlnet, error: null };
    },
    [controlnets, model?.base_model]
  );

  const recallControlNet = useCallback(
    (controlnetMetadataItem: ControlNetMetadataItem) => {
      const result = prepareControlNetMetadataItem(controlnetMetadataItem);

      if (!result.controlnet) {
        parameterNotSetToast(result.error);
        return;
      }

      dispatch(
        controlNetRecalled({
          ...result.controlnet,
        })
      );

      parameterSetToast();
    },
    [
      prepareControlNetMetadataItem,
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
        seed,
        steps,
        width,
        strength,
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

      dispatch(controlNetReset());
      dispatch(controlNetEnabled());
      controlnets?.forEach((controlnet) => {
        const result = prepareControlNetMetadataItem(controlnet);
        if (result.controlnet) {
          dispatch(controlNetRecalled(result.controlnet));
        }
      });

      allParameterSetToast();
    },
    [
      allParameterNotSetToast,
      allParameterSetToast,
      dispatch,
      prepareLoRAMetadataItem,
      prepareControlNetMetadataItem,
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
    recallSteps,
    recallWidth,
    recallHeight,
    recallStrength,
    recallLoRA,
    recallControlNet,
    recallAllParameters,
    sendToImageToImage,
  };
};
