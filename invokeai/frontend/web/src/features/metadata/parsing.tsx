import type { AppStore } from 'app/store/store';
import { useAppStore } from 'app/store/storeHooks';
import { withResultAsync } from 'common/util/result';
import { get } from 'es-toolkit/compat';
import { bboxHeightChanged, bboxWidthChanged } from 'features/controlLayers/store/canvasSlice';
import {
  negativePrompt2Changed,
  negativePromptChanged,
  positivePrompt2Changed,
  positivePromptChanged,
  refinerModelChanged,
  setCfgRescaleMultiplier,
  setCfgScale,
  setGuidance,
  setImg2imgStrength,
  setRefinerCFGScale,
  setRefinerNegativeAestheticScore,
  setRefinerPositiveAestheticScore,
  setRefinerScheduler,
  setRefinerStart,
  setRefinerSteps,
  setScheduler,
  setSeamlessXAxis,
  setSeamlessYAxis,
  setSeed,
  setSteps,
  vaeSelected,
} from 'features/controlLayers/store/paramsSlice';
import type { ModelIdentifierField } from 'features/nodes/types/common';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { zModelIdentifier } from 'features/nodes/types/v2/common';
import { modelSelected } from 'features/parameters/store/actions';
import type {
  ParameterCFGRescaleMultiplier,
  ParameterCFGScale,
  ParameterGuidance,
  ParameterHeight,
  ParameterModel,
  ParameterNegativePrompt,
  ParameterPositivePrompt,
  ParameterPositiveStylePromptSDXL,
  ParameterScheduler,
  ParameterSDXLRefinerModel,
  ParameterSDXLRefinerNegativeAestheticScore,
  ParameterSDXLRefinerPositiveAestheticScore,
  ParameterSDXLRefinerStart,
  ParameterSeamlessX,
  ParameterSeamlessY,
  ParameterSeed,
  ParameterSteps,
  ParameterStrength,
  ParameterVAEModel,
  ParameterWidth,
} from 'features/parameters/types/parameterSchemas';
import {
  zParameterCFGRescaleMultiplier,
  zParameterCFGScale,
  zParameterGuidance,
  zParameterImageDimension,
  zParameterNegativePrompt,
  zParameterNegativeStylePromptSDXL,
  zParameterPositivePrompt,
  zParameterPositiveStylePromptSDXL,
  zParameterScheduler,
  zParameterSDXLRefinerNegativeAestheticScore,
  zParameterSDXLRefinerPositiveAestheticScore,
  zParameterSDXLRefinerStart,
  zParameterSeamlessX,
  zParameterSeamlessY,
  zParameterSeed,
  zParameterSteps,
  zParameterStrength,
} from 'features/parameters/types/parameterSchemas';
import type { TFunction } from 'i18next';
import { type ReactNode, useCallback, useEffect, useState } from 'react';
import { modelsApi } from 'services/api/endpoints/models';
import type { ModelType } from 'services/api/types';
import { assert } from 'tsafe';
import z from 'zod/v4';

type UnparsedData = {
  isParsed: false;
  isSuccess: false;
  isError: false;
  value: null;
  error: null;
};

type ParsedSuccessData<T> = {
  isParsed: true;
  isSuccess: true;
  isError: false;
  value: T;
  error: null;
};

type ParsedErrorData = {
  isParsed: true;
  isSuccess: false;
  isError: true;
  value: null;
  error: Error;
};

export type Data<T> = UnparsedData | ParsedSuccessData<T> | ParsedErrorData;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
export type MetadataHandler<T = any> = {
  type: string;
  parse: (metadata: unknown, store: AppStore) => Promise<T> | T;
  recall?: (value: T, store: AppStore) => void;
  renderLabel: (value: T, t: TFunction) => ReactNode;
  renderValue: (value: T, t: TFunction) => ReactNode;
};

//#region Created By
const CreatedBy: MetadataHandler<string> = {
  type: 'CreatedBy',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'created_by');
    const parsed = z.string().parse(raw);
    return parsed;
  },
  renderLabel: (_value, t) => t('metadata.createdBy'),
  renderValue: (value) => value,
};
//#endregion Created By

//#region Generation Mode
const GenerationMode: MetadataHandler<string> = {
  type: 'GenerationMode',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'generation_mode');
    const parsed = z.string().parse(raw);
    return parsed;
  },
  renderLabel: (_value, t) => t('metadata.generationMode'),
  renderValue: (value) => value,
};
//#endregion Generation Mode

//#region Positive Prompt
const PositivePrompt: MetadataHandler<ParameterPositivePrompt> = {
  type: 'PositivePrompt',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'positive_prompt');
    const parsed = zParameterPositivePrompt.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(positivePromptChanged(value));
  },
  renderLabel: (_value, t) => t('metadata.positivePrompt'),
  renderValue: (value) => value,
};
//#endregion Positive Prompt

//#region Negative Prompt
const NegativePrompt: MetadataHandler<ParameterNegativePrompt> = {
  type: 'NegativePrompt',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'negative_prompt');
    const parsed = zParameterNegativePrompt.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(negativePromptChanged(value));
  },
  renderLabel: (_value, t) => t('metadata.negativePrompt'),
  renderValue: (value) => value,
};
//#endregion Negative Prompt

//#region SDXL Positive Style Prompt
const PositiveStylePrompt: MetadataHandler<ParameterPositiveStylePromptSDXL> = {
  type: 'PositiveStylePrompt',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'positive_style_prompt');
    const parsed = zParameterPositiveStylePromptSDXL.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(positivePrompt2Changed(value));
  },
  renderLabel: (_value, t) => t('sdxl.posStylePrompt'),
  renderValue: (value) => value,
};
//#endregion SDXL Positive Style Prompt

//#region SDXL Negative Style Prompt
const NegativeStylePrompt: MetadataHandler<ParameterPositiveStylePromptSDXL> = {
  type: 'NegativeStylePrompt',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'negative_style_prompt');
    const parsed = zParameterNegativeStylePromptSDXL.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(negativePrompt2Changed(value));
  },
  renderLabel: (_value, t) => t('sdxl.negStylePrompt'),
  renderValue: (value) => value,
};
//#endregion SDXL Negative Style Prompt

//#region CFG Scale
const CFGScale: MetadataHandler<ParameterCFGScale> = {
  type: 'CFGScale',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'cfg_scale');
    const parsed = zParameterCFGScale.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setCfgScale(value));
  },
  renderLabel: (_value, t) => t('metadata.cfgScale'),
  renderValue: (value) => value,
};
//#endregion CFG Scale

//#region CFG Rescale Multiplier
const CFGRescaleMultiplier: MetadataHandler<ParameterCFGRescaleMultiplier> = {
  type: 'CFGRescaleMultiplier',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'cfg_rescale_multiplier');
    const parsed = zParameterCFGRescaleMultiplier.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setCfgRescaleMultiplier(value));
  },
  renderLabel: (_value, t) => t('metadata.cfgRescaleMultiplier'),
  renderValue: (value) => value,
};
//#endregion CFG Rescale Multiplier

//#region Guidance
const Guidance: MetadataHandler<ParameterGuidance> = {
  type: 'Guidance',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'guidance');
    const parsed = zParameterGuidance.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setGuidance(value));
  },
  renderLabel: (_value, t) => t('metadata.guidance'),
  renderValue: (value) => value,
};
//#endregion Guidance

//#region Scheduler
const Scheduler: MetadataHandler<ParameterScheduler> = {
  type: 'Scheduler',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'scheduler');
    const parsed = zParameterScheduler.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setScheduler(value));
  },
  renderLabel: (_value, t) => t('metadata.scheduler'),
  renderValue: (value) => value,
};
//#endregion Scheduler

//#region Width
const Width: MetadataHandler<ParameterWidth> = {
  type: 'Width',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'width');
    const parsed = zParameterImageDimension.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(bboxWidthChanged({ width: value, updateAspectRatio: true, clamp: true }));
  },
  renderLabel: (_value, t) => t('metadata.width'),
  renderValue: (value) => value,
};
//#endregion Width

//#region Height
const Height: MetadataHandler<ParameterHeight> = {
  type: 'Height',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'height');
    const parsed = zParameterImageDimension.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(bboxHeightChanged({ height: value, updateAspectRatio: true, clamp: true }));
  },
  renderLabel: (_value, t) => t('metadata.height'),
  renderValue: (value) => value,
};
//#endregion Height

//#region Seed
const Seed: MetadataHandler<ParameterSeed> = {
  type: 'Seed',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'seed');
    const parsed = zParameterSeed.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setSeed(value));
  },
  renderLabel: (_value, t) => t('metadata.seed'),
  renderValue: (value) => value,
};
//#endregion Seed

//#region Steps
const Steps: MetadataHandler<ParameterSteps> = {
  type: 'Steps',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'steps');
    const parsed = zParameterSteps.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setSteps(value));
  },
  renderLabel: (_value, t) => t('metadata.steps'),
  renderValue: (value) => value,
};
//#endregion Steps

//#region DenoisingStrength
const DenoisingStrength: MetadataHandler<ParameterStrength> = {
  type: 'DenoisingStrength',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'strength');
    const parsed = zParameterStrength.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setImg2imgStrength(value));
  },
  renderLabel: (_value, t) => t('metadata.strength'),
  renderValue: (value) => value,
};
//#endregion DenoisingStrength

//#region SeamlessX
const SeamlessX: MetadataHandler<ParameterSeamlessX> = {
  type: 'SeamlessX',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'seamless_x');
    const parsed = zParameterSeamlessX.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setSeamlessXAxis(value));
  },
  renderLabel: (_value, t) => t('metadata.seamlessXAxis'),
  renderValue: (value) => value,
};
//#endregion SeamlessX

//#region SeamlessY
const SeamlessY: MetadataHandler<ParameterSeamlessY> = {
  type: 'SeamlessY',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'seamless_y');
    const parsed = zParameterSeamlessY.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setSeamlessYAxis(value));
  },
  renderLabel: (_value, t) => t('metadata.seamlessYAxis'),
  renderValue: (value) => value,
};
//#endregion SeamlessY

//#region RefinerModel
const RefinerModel: MetadataHandler<ParameterSDXLRefinerModel> = {
  type: 'RefinerModel',
  parse: async (metadata, store) => {
    const raw = get(metadata, 'refiner_model');
    const parsed = await parseModelIdentifier(raw, store, 'main');
    assert(parsed.type === 'main');
    assert(parsed.base === 'sdxl-refiner');
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(refinerModelChanged(value));
  },
  renderLabel: (_value, t) => t('sdxl.refinermodel'),
  renderValue: (value) => `${value.name} (${value.base.toUpperCase()})`,
};
//#endregion RefinerModel

//#region RefinerSteps
const RefinerSteps: MetadataHandler<ParameterSteps> = {
  type: 'RefinerSteps',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'refiner_steps');
    const parsed = zParameterSteps.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setRefinerSteps(value));
  },
  renderLabel: (_value, t) => t('sdxl.refinerSteps'),
  renderValue: (value) => value,
};
//#endregion RefinerSteps

//#region RefinerCFGScale
const RefinerCFGScale: MetadataHandler<ParameterSteps> = {
  type: 'RefinerCFGScale',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'refiner_cfg_scale');
    const parsed = zParameterCFGScale.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setRefinerCFGScale(value));
  },
  renderLabel: (_value, t) => t('sdxl.cfgScale'),
  renderValue: (value) => value,
};
//#endregion RefinerCFGScale

//#region RefinerScheduler
const RefinerScheduler: MetadataHandler<ParameterScheduler> = {
  type: 'RefinerScheduler',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'refiner_scheduler');
    const parsed = zParameterScheduler.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setRefinerScheduler(value));
  },
  renderLabel: (_value, t) => t('sdxl.scheduler'),
  renderValue: (value) => value,
};
//#endregion RefinerScheduler

//#region RefinerPositiveAestheticScore
const RefinerPositiveAestheticScore: MetadataHandler<ParameterSDXLRefinerPositiveAestheticScore> = {
  type: 'RefinerPositiveAestheticScore',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'refiner_positive_aesthetic_score');
    const parsed = zParameterSDXLRefinerPositiveAestheticScore.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setRefinerPositiveAestheticScore(value));
  },
  renderLabel: (_value, t) => t('sdxl.posAestheticScore'),
  renderValue: (value) => value,
};
//#endregion RefinerPositiveAestheticScore

//#region RefinerNegativeAestheticScore
const RefinerNegativeAestheticScore: MetadataHandler<ParameterSDXLRefinerNegativeAestheticScore> = {
  type: 'RefinerNegativeAestheticScore',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'refiner_negative_aesthetic_score');
    const parsed = zParameterSDXLRefinerNegativeAestheticScore.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setRefinerNegativeAestheticScore(value));
  },
  renderLabel: (_value, t) => t('sdxl.negAestheticScore'),
  renderValue: (value) => value,
};
//#endregion RefinerNegativeAestheticScore

//#region RefinerDenoisingStart
const RefinerDenoisingStart: MetadataHandler<ParameterSDXLRefinerStart> = {
  type: 'RefinerDenoisingStart',
  parse: (metadata, _store) => {
    const raw = get(metadata, 'refiner_start');
    const parsed = zParameterSDXLRefinerStart.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setRefinerStart(value));
  },
  renderLabel: (_value, t) => t('sdxl.refinerStart'),
  renderValue: (value) => value,
};
//#endregion RefinerDenoisingStart

//#region MainModel
const MainModel: MetadataHandler<ParameterModel> = {
  type: 'MainModel',
  parse: async (metadata, store) => {
    const raw = get(metadata, 'model');
    const parsed = await parseModelIdentifier(raw, store, 'main');
    assert(parsed.type === 'main');
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(modelSelected(value));
  },
  renderLabel: (_value, t) => t('metadata.model'),
  renderValue: (value) => `${value.name} (${value.base.toUpperCase()})`,
};
//#endregion MainModel

//#region VAEModel
const VAEModel: MetadataHandler<ParameterVAEModel> = {
  type: 'VAEModel',
  parse: async (metadata, store) => {
    const raw = get(metadata, 'vae');
    const parsed = await parseModelIdentifier(raw, store, 'vae');
    assert(parsed.type === 'vae');
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(vaeSelected(value));
  },
  renderLabel: (_value, t) => t('metadata.vae'),
  renderValue: (value) => `${value.name} (${value.base.toUpperCase()})`,
};
//#endregion VAEModel

export const MetadataHanders = {
  CreatedBy,
  GenerationMode,
  PositivePrompt,
  NegativePrompt,
  PositiveStylePrompt,
  NegativeStylePrompt,
  CFGScale,
  CFGRescaleMultiplier,
  Guidance,
  Scheduler,
  Width,
  Height,
  Seed,
  Steps,
  DenoisingStrength,
  SeamlessX,
  SeamlessY,
  RefinerModel,
  RefinerSteps,
  RefinerCFGScale,
  RefinerScheduler,
  RefinerPositiveAestheticScore,
  RefinerNegativeAestheticScore,
  RefinerDenoisingStart,
  MainModel,
  VAEModel,
} satisfies Record<string, MetadataHandler>;

export function useMetadata<T>(metadata: unknown, handler: MetadataHandler<T>) {
  const store = useAppStore();
  const [data, setData] = useState<Data<T>>(() => ({
    isParsed: false,
    isSuccess: false,
    isError: false,
    value: null,
    error: null,
  }));

  const parse = useCallback(
    async (metadata: unknown) => {
      const result = await withResultAsync(async () => await Promise.resolve(handler.parse(metadata, store)));
      if (result.isOk()) {
        setData({
          isParsed: true,
          isSuccess: true,
          isError: false,
          value: result.value,
          error: null,
        });
      } else {
        setData({
          isParsed: true,
          isSuccess: false,
          isError: true,
          value: null,
          error: result.error,
        });
      }
    },
    [handler, store]
  );

  useEffect(() => {
    parse(metadata);
  }, [metadata, parse]);

  const recall = useCallback(() => {
    if (!data.isSuccess) {
      return;
    }
    handler.recall?.(data.value, store);
  }, [data.isSuccess, data.value, handler, store]);

  return { data, recall };
}

const parseModelIdentifier = async (raw: unknown, store: AppStore, type: ModelType): Promise<ModelIdentifierField> => {
  // First try the current format identifier: key, name, base, type, hash
  try {
    const identifier = zModelIdentifierField.parse(raw);
    const modelConfig = store
      .dispatch(modelsApi.endpoints.getModelConfig.initiate(identifier.key, { subscribe: false }))
      .unwrap();
    return zModelIdentifierField.parse(modelConfig);
  } catch {
    // noop
  }
  // Fall back to old format identifier: model_name, base_model
  try {
    const identifier = zModelIdentifier.parse(raw);
    const modelConfig = await store
      .dispatch(
        modelsApi.endpoints.getModelConfigByAttrs.initiate(
          { name: identifier.model_name, base: identifier.base_model, type },
          { subscribe: false }
        )
      )
      .unwrap();
    return zModelIdentifierField.parse(modelConfig);
  } catch {
    // noop
  }
  throw new Error('Unable to parse model identifier');
};
