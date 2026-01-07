/* eslint-disable @typescript-eslint/no-explicit-any */
import { Text } from '@invoke-ai/ui-library';
import type { AppStore } from 'app/store/store';
import { useAppStore } from 'app/store/storeHooks';
import { WrappedError } from 'common/util/result';
import { get, isArray, isString } from 'es-toolkit/compat';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { bboxHeightChanged, bboxWidthChanged, canvasMetadataRecalled } from 'features/controlLayers/store/canvasSlice';
import { loraAllDeleted, loraRecalled } from 'features/controlLayers/store/lorasSlice';
import {
  heightChanged,
  negativePromptChanged,
  positivePromptChanged,
  refinerModelChanged,
  selectBase,
  setCfgRescaleMultiplier,
  setCfgScale,
  setClipSkip,
  setFluxScheduler,
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
  setZImageScheduler,
  vaeSelected,
  widthChanged,
  zImageQwen3EncoderModelSelected,
  zImageQwen3SourceModelSelected,
  zImageVaeModelSelected,
} from 'features/controlLayers/store/paramsSlice';
import { refImagesRecalled } from 'features/controlLayers/store/refImagesSlice';
import type { CanvasMetadata, LoRA, RefImageState } from 'features/controlLayers/store/types';
import { zCanvasMetadata, zCanvasReferenceImageState_OLD, zRefImageState } from 'features/controlLayers/store/types';
import type { ModelIdentifierField, ModelType } from 'features/nodes/types/common';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { zModelIdentifier } from 'features/nodes/types/v2/common';
import { modelSelected } from 'features/parameters/store/actions';
import type {
  ParameterCFGRescaleMultiplier,
  ParameterCFGScale,
  ParameterCLIPSkip,
  ParameterGuidance,
  ParameterHeight,
  ParameterModel,
  ParameterNegativePrompt,
  ParameterPositivePrompt,
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
  zLoRAWeight,
  zParameterCFGRescaleMultiplier,
  zParameterCFGScale,
  zParameterCLIPSkip,
  zParameterGuidance,
  zParameterImageDimension,
  zParameterNegativePrompt,
  zParameterPositivePrompt,
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
import { toast } from 'features/toast/toast';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { t } from 'i18next';
import type { ComponentType } from 'react';
import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { imagesApi } from 'services/api/endpoints/images';
import { modelsApi } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';
import { assert } from 'tsafe';
import z from 'zod';

const MetadataLabel = ({ i18nKey }: { i18nKey: string }) => {
  const { t } = useTranslation();
  return (
    <Text as="span" fontWeight="semibold" whiteSpace="pre-wrap" me={2}>
      {t(i18nKey)}:
    </Text>
  );
};

const MetadataPrimitiveValue = ({ value }: { value: string | number | boolean | null | undefined }) => {
  if (value === null || value === undefined) {
    return null;
  }
  if (isString(value)) {
    return <Text as="span">{value || '<empty string>'}</Text>;
  }
  return <Text as="span">{String(value)}</Text>;
};

const getProperty = (obj: unknown, path: string): unknown => {
  return get(obj, path) as unknown;
};

type UnparsedData = {
  isParsed: false;
  isSuccess: false;
  isError: false;
  value: null;
  error: null;
};
const buildUnparsedData = (): UnparsedData => ({
  isParsed: false,
  isSuccess: false,
  isError: false,
  value: null,
  error: null,
});

export type ParsedSuccessData<T> = {
  isParsed: true;
  isSuccess: true;
  isError: false;
  value: T;
  error: null;
};
const buildParsedSuccessData = <T,>(value: T): ParsedSuccessData<T> => ({
  isParsed: true,
  isSuccess: true,
  isError: false,
  value,
  error: null,
});

type ParsedErrorData = {
  isParsed: true;
  isSuccess: false;
  isError: true;
  value: null;
  error: Error;
};
const buildParsedErrorData = (error: Error): ParsedErrorData => ({
  isParsed: true,
  isSuccess: false,
  isError: true,
  value: null,
  error,
});

type Data<T> = UnparsedData | ParsedSuccessData<T> | ParsedErrorData;

const SingleMetadataKey = Symbol('SingleMetadataKey');
type SingleMetadataValueProps<T> = {
  value: T;
};
export type SingleMetadataHandler<T> = {
  [SingleMetadataKey]: true;
  type: string;
  parse: (metadata: unknown, store: AppStore) => Promise<T>;
  recall: (value: T, store: AppStore) => void;
  i18nKey: string;
  LabelComponent: ComponentType<{ i18nKey: string }>;
  ValueComponent: ComponentType<SingleMetadataValueProps<T>>;
};

const CollectionMetadataKey = Symbol('CollectionMetadataKey');
type CollectionMetadataValueProps<T extends any[]> = {
  value: T[number];
};
export type CollectionMetadataHandler<T extends any[]> = {
  [CollectionMetadataKey]: true;
  type: string;
  parse: (metadata: unknown, store: AppStore) => Promise<T>;
  recall: (values: T, store: AppStore) => void;
  recallOne: (value: T[number], store: AppStore) => void;
  i18nKey: string;
  LabelComponent: ComponentType<{ i18nKey: string }>;
  ValueComponent: ComponentType<CollectionMetadataValueProps<T>>;
};

const UnrecallableMetadataKey = Symbol('UnrecallableMetadataKey');
type UnrecallableMetadataValueProps<T> = {
  value: T;
};
export type UnrecallableMetadataHandler<T> = {
  [UnrecallableMetadataKey]: true;
  type: string;
  parse: (metadata: unknown, store: AppStore) => Promise<T>;
  i18nKey: string;
  LabelComponent: ComponentType<{ i18nKey: string }>;
  ValueComponent: ComponentType<UnrecallableMetadataValueProps<T>>;
};

const isSingleMetadataHandler = (
  handler: SingleMetadataHandler<any> | CollectionMetadataHandler<any[]> | UnrecallableMetadataHandler<any>
): handler is SingleMetadataHandler<any> => {
  return SingleMetadataKey in handler && handler[SingleMetadataKey] === true;
};

const isCollectionMetadataHandler = (
  handler: SingleMetadataHandler<any> | CollectionMetadataHandler<any[]> | UnrecallableMetadataHandler<any>
): handler is CollectionMetadataHandler<any[]> => {
  return CollectionMetadataKey in handler && handler[CollectionMetadataKey] === true;
};

//#region Created By
const CreatedBy: UnrecallableMetadataHandler<string> = {
  [UnrecallableMetadataKey]: true,
  type: 'CreatedBy',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'created_by');
    const parsed = z.string().parse(raw);
    return Promise.resolve(parsed);
  },
  i18nKey: 'metadata.createdBy',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: UnrecallableMetadataValueProps<string>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Created By

//#region Generation Mode
const GenerationMode: UnrecallableMetadataHandler<string> = {
  [UnrecallableMetadataKey]: true,
  type: 'GenerationMode',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'generation_mode');
    const parsed = z.string().parse(raw);
    return Promise.resolve(parsed);
  },
  i18nKey: 'metadata.generationMode',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: UnrecallableMetadataValueProps<string>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Generation Mode

//#region Positive Prompt
const PositivePrompt: SingleMetadataHandler<ParameterPositivePrompt> = {
  [SingleMetadataKey]: true,
  type: 'PositivePrompt',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'positive_prompt');
    const parsed = zParameterPositivePrompt.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(positivePromptChanged(value));
  },
  i18nKey: 'metadata.positivePrompt',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterPositivePrompt>) => (
    <MetadataPrimitiveValue value={value} />
  ),
};
//#endregion Positive Prompt

//#region Negative Prompt
const NegativePrompt: SingleMetadataHandler<ParameterNegativePrompt> = {
  [SingleMetadataKey]: true,
  type: 'NegativePrompt',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'negative_prompt');
    const parsed = zParameterNegativePrompt.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(negativePromptChanged(value || null));
  },
  i18nKey: 'metadata.negativePrompt',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterNegativePrompt>) => (
    <MetadataPrimitiveValue value={value} />
  ),
};
//#endregion Negative Prompt

//#region CFG Scale
const CFGScale: SingleMetadataHandler<ParameterCFGScale> = {
  [SingleMetadataKey]: true,
  type: 'CFGScale',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'cfg_scale');
    const parsed = zParameterCFGScale.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setCfgScale(value));
  },
  i18nKey: 'metadata.cfgScale',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterCFGScale>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion CFG Scale

//#region CFG Rescale Multiplier
const CFGRescaleMultiplier: SingleMetadataHandler<ParameterCFGRescaleMultiplier> = {
  [SingleMetadataKey]: true,
  type: 'CFGRescaleMultiplier',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'cfg_rescale_multiplier');
    const parsed = zParameterCFGRescaleMultiplier.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setCfgRescaleMultiplier(value));
  },
  i18nKey: 'metadata.cfgRescaleMultiplier',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterCFGRescaleMultiplier>) => (
    <MetadataPrimitiveValue value={value} />
  ),
};
//#endregion CFG Rescale Multiplier

//#region CLIP Skip
const CLIPSkip: SingleMetadataHandler<ParameterCLIPSkip> = {
  [SingleMetadataKey]: true,
  type: 'CLIPSkip',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'clip_skip');
    const parsed = zParameterCLIPSkip.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setClipSkip(value));
  },
  i18nKey: 'metadata.clipSkip',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterCLIPSkip>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion CLIP Skip

//#region Guidance
const Guidance: SingleMetadataHandler<ParameterGuidance> = {
  [SingleMetadataKey]: true,
  type: 'Guidance',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'guidance');
    const parsed = zParameterGuidance.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setGuidance(value));
  },
  i18nKey: 'metadata.guidance',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterGuidance>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Guidance

//#region Scheduler
const Scheduler: SingleMetadataHandler<ParameterScheduler> = {
  [SingleMetadataKey]: true,
  type: 'Scheduler',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'scheduler');
    const parsed = zParameterScheduler.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    // Dispatch to the appropriate scheduler based on the current model base
    const base = selectBase(store.getState());
    if (base === 'flux') {
      // Flux only supports euler, heun, lcm
      if (value === 'euler' || value === 'heun' || value === 'lcm') {
        store.dispatch(setFluxScheduler(value));
      }
    } else if (base === 'z-image') {
      // Z-Image only supports euler, heun, lcm
      if (value === 'euler' || value === 'heun' || value === 'lcm') {
        store.dispatch(setZImageScheduler(value));
      }
    } else {
      // SD, SDXL, SD3, CogView4, etc. use the general scheduler
      store.dispatch(setScheduler(value));
    }
  },
  i18nKey: 'metadata.scheduler',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterScheduler>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Scheduler

//#region Width
const Width: SingleMetadataHandler<ParameterWidth> = {
  [SingleMetadataKey]: true,
  type: 'Width',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'width');
    const parsed = zParameterImageDimension.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    const activeTab = selectActiveTab(store.getState());
    if (activeTab === 'canvas') {
      store.dispatch(bboxWidthChanged({ width: value, updateAspectRatio: true, clamp: true }));
    } else if (activeTab === 'generate') {
      store.dispatch(widthChanged({ width: value, updateAspectRatio: true, clamp: true }));
    }
  },
  i18nKey: 'metadata.width',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterWidth>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Width

//#region Height
const Height: SingleMetadataHandler<ParameterHeight> = {
  [SingleMetadataKey]: true,
  type: 'Height',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'height');
    const parsed = zParameterImageDimension.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    const activeTab = selectActiveTab(store.getState());
    if (activeTab === 'canvas') {
      store.dispatch(bboxHeightChanged({ height: value, updateAspectRatio: true, clamp: true }));
    } else if (activeTab === 'generate') {
      store.dispatch(heightChanged({ height: value, updateAspectRatio: true, clamp: true }));
    }
  },
  i18nKey: 'metadata.height',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterHeight>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Height

//#region Seed
const Seed: SingleMetadataHandler<ParameterSeed> = {
  [SingleMetadataKey]: true,
  type: 'Seed',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'seed');
    const parsed = zParameterSeed.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setSeed(value));
  },
  i18nKey: 'metadata.seed',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterSeed>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Seed

//#region Steps
const Steps: SingleMetadataHandler<ParameterSteps> = {
  [SingleMetadataKey]: true,
  type: 'Steps',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'steps');
    const parsed = zParameterSteps.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setSteps(value));
  },
  i18nKey: 'metadata.steps',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterSteps>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Steps

//#region DenoisingStrength
const DenoisingStrength: SingleMetadataHandler<ParameterStrength> = {
  [SingleMetadataKey]: true,
  type: 'DenoisingStrength',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'strength');
    const parsed = zParameterStrength.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setImg2imgStrength(value));
  },
  i18nKey: 'metadata.strength',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterStrength>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion DenoisingStrength

//#region SeamlessX
const SeamlessX: SingleMetadataHandler<ParameterSeamlessX> = {
  [SingleMetadataKey]: true,
  type: 'SeamlessX',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'seamless_x');
    const parsed = zParameterSeamlessX.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setSeamlessXAxis(value));
  },
  i18nKey: 'metadata.seamlessXAxis',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterSeamlessX>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion SeamlessX

//#region SeamlessY
const SeamlessY: SingleMetadataHandler<ParameterSeamlessY> = {
  [SingleMetadataKey]: true,
  type: 'SeamlessY',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'seamless_y');
    const parsed = zParameterSeamlessY.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setSeamlessYAxis(value));
  },
  i18nKey: 'metadata.seamlessYAxis',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterSeamlessY>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion SeamlessY

//#region RefinerModel
const RefinerModel: SingleMetadataHandler<ParameterSDXLRefinerModel> = {
  [SingleMetadataKey]: true,
  type: 'RefinerModel',
  parse: async (metadata, store) => {
    const raw = getProperty(metadata, 'refiner_model');
    const parsed = await parseModelIdentifier(raw, store, 'main');
    assert(parsed.type === 'main');
    assert(parsed.base === 'sdxl-refiner');
    assert(isCompatibleWithMainModel(parsed, store));
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(refinerModelChanged(value));
  },
  i18nKey: 'sdxl.refinermodel',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterSDXLRefinerModel>) => (
    <MetadataPrimitiveValue value={`${value.name} (${value.base.toUpperCase()})`} />
  ),
};
//#endregion RefinerModel

//#region RefinerSteps
const RefinerSteps: SingleMetadataHandler<ParameterSteps> = {
  [SingleMetadataKey]: true,
  type: 'RefinerSteps',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'refiner_steps');
    const parsed = zParameterSteps.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setRefinerSteps(value));
  },
  i18nKey: 'sdxl.refinerSteps',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterSteps>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion RefinerSteps

//#region RefinerCFGScale
const RefinerCFGScale: SingleMetadataHandler<ParameterCFGScale> = {
  [SingleMetadataKey]: true,
  type: 'RefinerCFGScale',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'refiner_cfg_scale');
    const parsed = zParameterCFGScale.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setRefinerCFGScale(value));
  },
  i18nKey: 'sdxl.cfgScale',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterCFGScale>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion RefinerCFGScale

//#region RefinerScheduler
const RefinerScheduler: SingleMetadataHandler<ParameterScheduler> = {
  [SingleMetadataKey]: true,
  type: 'RefinerScheduler',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'refiner_scheduler');
    const parsed = zParameterScheduler.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setRefinerScheduler(value));
  },
  i18nKey: 'sdxl.scheduler',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterScheduler>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion RefinerScheduler

//#region RefinerPositiveAestheticScore
const RefinerPositiveAestheticScore: SingleMetadataHandler<ParameterSDXLRefinerPositiveAestheticScore> = {
  [SingleMetadataKey]: true,
  type: 'RefinerPositiveAestheticScore',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'refiner_positive_aesthetic_score');
    const parsed = zParameterSDXLRefinerPositiveAestheticScore.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setRefinerPositiveAestheticScore(value));
  },
  i18nKey: 'sdxl.posAestheticScore',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterSDXLRefinerPositiveAestheticScore>) => (
    <MetadataPrimitiveValue value={value} />
  ),
};
//#endregion RefinerPositiveAestheticScore

//#region RefinerNegativeAestheticScore
const RefinerNegativeAestheticScore: SingleMetadataHandler<ParameterSDXLRefinerNegativeAestheticScore> = {
  [SingleMetadataKey]: true,
  type: 'RefinerNegativeAestheticScore',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'refiner_negative_aesthetic_score');
    const parsed = zParameterSDXLRefinerNegativeAestheticScore.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setRefinerNegativeAestheticScore(value));
  },
  i18nKey: 'sdxl.negAestheticScore',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterSDXLRefinerNegativeAestheticScore>) => (
    <MetadataPrimitiveValue value={value} />
  ),
};
//#endregion RefinerNegativeAestheticScore

//#region RefinerDenoisingStart
const RefinerDenoisingStart: SingleMetadataHandler<ParameterSDXLRefinerStart> = {
  [SingleMetadataKey]: true,
  type: 'RefinerDenoisingStart',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'refiner_start');
    const parsed = zParameterSDXLRefinerStart.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setRefinerStart(value));
  },
  i18nKey: 'sdxl.refinerStart',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterSDXLRefinerStart>) => (
    <MetadataPrimitiveValue value={value} />
  ),
};
//#endregion RefinerDenoisingStart

//#region MainModel
const MainModel: SingleMetadataHandler<ParameterModel> = {
  [SingleMetadataKey]: true,
  type: 'MainModel',
  parse: async (metadata, store) => {
    const raw = getProperty(metadata, 'model');
    const parsed = await parseModelIdentifier(raw, store, 'main');
    assert(parsed.type === 'main');
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(modelSelected(value));
  },
  i18nKey: 'metadata.model',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterModel>) => (
    <MetadataPrimitiveValue value={`${value.name} (${value.base.toUpperCase()})`} />
  ),
};
//#endregion MainModel

//#region VAEModel
const VAEModel: SingleMetadataHandler<ParameterVAEModel> = {
  [SingleMetadataKey]: true,
  type: 'VAEModel',
  parse: async (metadata, store) => {
    const raw = getProperty(metadata, 'vae');
    const parsed = await parseModelIdentifier(raw, store, 'vae');
    assert(parsed.type === 'vae');
    assert(isCompatibleWithMainModel(parsed, store));
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(vaeSelected(value));
  },
  i18nKey: 'metadata.vae',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterVAEModel>) => (
    <MetadataPrimitiveValue value={`${value.name} (${value.base.toUpperCase()})`} />
  ),
};
//#endregion VAEModel

//#region Qwen3EncoderModel
const Qwen3EncoderModel: SingleMetadataHandler<ModelIdentifierField> = {
  [SingleMetadataKey]: true,
  type: 'Qwen3EncoderModel',
  parse: async (metadata, store) => {
    const raw = getProperty(metadata, 'qwen3_encoder');
    const parsed = await parseModelIdentifier(raw, store, 'qwen3_encoder');
    assert(parsed.type === 'qwen3_encoder');
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    // Clear conflicting Qwen3Source when setting Encoder (mutually exclusive)
    store.dispatch(zImageQwen3SourceModelSelected(null));
    store.dispatch(zImageQwen3EncoderModelSelected(value));
  },
  i18nKey: 'metadata.qwen3Encoder',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ModelIdentifierField>) => (
    <MetadataPrimitiveValue value={`${value.name} (${value.base.toUpperCase()})`} />
  ),
};
//#endregion Qwen3EncoderModel

//#region ZImageVAEModel
const ZImageVAEModel: SingleMetadataHandler<ModelIdentifierField> = {
  [SingleMetadataKey]: true,
  type: 'ZImageVAEModel',
  parse: async (metadata, store) => {
    const raw = getProperty(metadata, 'vae');
    const parsed = await parseModelIdentifier(raw, store, 'vae');
    assert(parsed.type === 'vae');
    // Only recall if the current main model is Z-Image
    const base = selectBase(store.getState());
    assert(base === 'z-image', 'ZImageVAEModel handler only works with Z-Image models');
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    // Clear conflicting Qwen3Source when setting VAE (mutually exclusive)
    store.dispatch(zImageQwen3SourceModelSelected(null));
    store.dispatch(zImageVaeModelSelected(value));
  },
  i18nKey: 'metadata.vae',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ModelIdentifierField>) => (
    <MetadataPrimitiveValue value={`${value.name} (${value.base.toUpperCase()})`} />
  ),
};
//#endregion ZImageVAEModel

//#region ZImageQwen3SourceModel
const ZImageQwen3SourceModel: SingleMetadataHandler<ModelIdentifierField> = {
  [SingleMetadataKey]: true,
  type: 'ZImageQwen3SourceModel',
  parse: async (metadata, store) => {
    const raw = getProperty(metadata, 'qwen3_source');
    const parsed = await parseModelIdentifier(raw, store, 'main');
    assert(parsed.type === 'main');
    // Only recall if the current main model is Z-Image
    const base = selectBase(store.getState());
    assert(base === 'z-image', 'ZImageQwen3SourceModel handler only works with Z-Image models');
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    // Clear conflicting VAE and Encoder when setting Qwen3Source (mutually exclusive)
    store.dispatch(zImageVaeModelSelected(null));
    store.dispatch(zImageQwen3EncoderModelSelected(null));
    store.dispatch(zImageQwen3SourceModelSelected(value));
  },
  i18nKey: 'metadata.qwen3Source',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ModelIdentifierField>) => (
    <MetadataPrimitiveValue value={`${value.name} (${value.base.toUpperCase()})`} />
  ),
};
//#endregion ZImageQwen3SourceModel

//#region LoRAs
const LoRAs: CollectionMetadataHandler<LoRA[]> = {
  [CollectionMetadataKey]: true,
  type: 'LoRAs',
  parse: async (metadata, store) => {
    const rawArray = getProperty(metadata, 'loras');

    if (!rawArray) {
      return [];
    }

    assert(isArray(rawArray));

    const loras: LoRA[] = [];

    for (const rawItem of rawArray) {
      try {
        let identifier: ModelIdentifierField | null = null;

        try {
          // New format - { model: ModelIdenfifierField }
          const rawIdentifier = getProperty(rawItem, 'model');
          identifier = await parseModelIdentifier(rawIdentifier, store, 'lora');
        } catch {
          // Old format - { lora : { key: string } }
          const key = getProperty(rawItem, 'lora.key');
          assert(isString(key));
          // No need to catch here - if this throws, we move on to the next item
          identifier = await getModelIdentiferFromKey(key, store);
        }

        assert(identifier.type === 'lora');
        assert(isCompatibleWithMainModel(identifier, store));

        const weight = getProperty(rawItem, 'weight');

        loras.push({
          id: getPrefixedId('lora'),
          model: identifier,
          weight: zLoRAWeight.parse(weight),
          isEnabled: true,
        });
      } catch {
        continue;
      }
    }

    if (loras.length > 0) {
      return loras;
    }

    throw new Error('No valid LoRAs found in metadata');
  },
  recallOne: (value, store) => {
    store.dispatch(loraRecalled({ lora: value }));
  },
  recall: (values, store) => {
    store.dispatch(loraAllDeleted());
    for (const lora of values) {
      store.dispatch(loraRecalled({ lora }));
    }
  },
  i18nKey: 'models.lora',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: CollectionMetadataValueProps<LoRA[]>) => (
    <MetadataPrimitiveValue value={`${value.model.name} (${value.model.base.toUpperCase()}) - ${value.weight}`} />
  ),
};
//#endregion LoRAs

//#region CanvasLayers
const CanvasLayers: SingleMetadataHandler<CanvasMetadata> = {
  [SingleMetadataKey]: true,
  type: 'CanvasLayers',
  parse: async (metadata, store) => {
    const raw = getProperty(metadata, 'canvas_v2_metadata');
    // This validator fetches all referenced images. If any do not exist, validation fails. The logic for this is in
    // the zImageWithDims schema.
    const parsed = await zCanvasMetadata.parseAsync(raw);

    for (const entity of parsed.controlLayers) {
      if (entity.controlAdapter.model) {
        await throwIfModelDoesNotExist(entity.controlAdapter.model.key, store);
      }
      for (const object of entity.objects) {
        if (object.type === 'image' && 'image_name' in object.image) {
          await throwIfImageDoesNotExist(object.image.image_name, store);
        }
      }
    }

    for (const entity of parsed.inpaintMasks) {
      for (const object of entity.objects) {
        if (object.type === 'image' && 'image_name' in object.image) {
          await throwIfImageDoesNotExist(object.image.image_name, store);
        }
      }
    }

    for (const entity of parsed.rasterLayers) {
      for (const object of entity.objects) {
        if (object.type === 'image' && 'image_name' in object.image) {
          await throwIfImageDoesNotExist(object.image.image_name, store);
        }
      }
    }

    for (const entity of parsed.regionalGuidance) {
      for (const object of entity.objects) {
        if (object.type === 'image' && 'image_name' in object.image) {
          await throwIfImageDoesNotExist(object.image.image_name, store);
        }
      }
      for (const refImage of entity.referenceImages) {
        if (refImage.config.image) {
          await throwIfImageDoesNotExist(refImage.config.image.image_name, store);
        }
        if (refImage.config.model) {
          await throwIfModelDoesNotExist(refImage.config.model.key, store);
        }
      }
    }

    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    if (
      value.controlLayers.length === 0 &&
      value.rasterLayers.length === 0 &&
      value.inpaintMasks.length === 0 &&
      value.regionalGuidance.length === 0
    ) {
      // Nothing to recall
      return;
    }
    store.dispatch(canvasMetadataRecalled(value));
  },
  i18nKey: 'metadata.canvasV2Metadata',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<CanvasMetadata>) => {
    const { t } = useTranslation();
    const count =
      value.controlLayers.length +
      value.rasterLayers.length +
      value.inpaintMasks.length +
      value.regionalGuidance.length;
    return <MetadataPrimitiveValue value={`${count} ${t('controlLayers.layer', { count })}`} />;
  },
};
//#endregion CanvasLayers

//#region RefImages
const RefImages: CollectionMetadataHandler<RefImageState[]> = {
  [CollectionMetadataKey]: true,
  type: 'RefImages',
  parse: async (metadata, store) => {
    let parsed: RefImageState[] | null = null;
    try {
      // First attempt to parse from the v6 slot
      const raw = getProperty(metadata, 'ref_images');
      parsed = z.array(zRefImageState).parse(raw);
    } catch {
      // Fall back to extracting from canvas metadata]
      const raw = getProperty(metadata, 'canvas_v2_metadata.referenceImages.entities');
      // This validator fetches all referenced images. If any do not exist, validation fails. The logic for this is in
      // the zImageWithDims schema.
      const oldParsed = await z.array(zCanvasReferenceImageState_OLD).parseAsync(raw);
      parsed = oldParsed.map(({ id, ipAdapter, isEnabled }) => ({
        id,
        config: ipAdapter,
        isEnabled,
      }));
    }

    if (!parsed) {
      throw new Error('No valid reference images found in metadata');
    }

    for (const refImage of parsed) {
      if (refImage.config.image) {
        await throwIfImageDoesNotExist(refImage.config.image.original.image.image_name, store);
      }
      if (refImage.config.model) {
        await throwIfModelDoesNotExist(refImage.config.model.key, store);
      }
    }

    return parsed;
  },
  recall: (value, store) => {
    const entities = value.map((data) => ({ ...data, id: getPrefixedId('reference_image') }));
    store.dispatch(refImagesRecalled({ entities, replace: true }));
  },
  recallOne: (data, store) => {
    const entities = [{ ...data, id: getPrefixedId('reference_image') }];
    store.dispatch(refImagesRecalled({ entities, replace: false }));
  },
  i18nKey: 'controlLayers.referenceImage',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: CollectionMetadataValueProps<RefImageState[]>) => {
    if (value.config.model) {
      return <MetadataPrimitiveValue value={value.config.model.name} />;
    }
    return <MetadataPrimitiveValue value="No model" />;
  },
};
//#endregion RefImages

export const ImageMetadataHandlers = {
  CreatedBy,
  GenerationMode,
  PositivePrompt,
  NegativePrompt,
  CFGScale,
  CFGRescaleMultiplier,
  CLIPSkip,
  Guidance,
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
  // Scheduler must be after MainModel so that base-dependent logic (z-image scheduler) works correctly
  Scheduler,
  VAEModel,
  Qwen3EncoderModel,
  ZImageVAEModel,
  ZImageQwen3SourceModel,
  LoRAs,
  CanvasLayers,
  RefImages,
  // TODO: These had parsers in the prev implementation, but they were never actually used?
  // controlNet: parseControlNet,
  // controlNets: parseAllControlNets,
  // t2iAdapter: parseT2IAdapter,
  // t2iAdapters: parseAllT2IAdapters,
  // ipAdapter: parseIPAdapter,
  // ipAdapters: parseAllIPAdapters,
  // controlNetToControlLayer: parseControlNetToControlAdapterLayer,
  // t2iAdapterToControlAdapterLayer: parseT2IAdapterToControlAdapterLayer,
  // ipAdapterToIPAdapterLayer: parseIPAdapterToIPAdapterLayer,
} as const;

const successToast = (parameter: string) => {
  toast({
    id: 'PARAMETER_SET',
    title: t('toast.parameterSet'),
    description: t('toast.parameterSetDesc', { parameter }),
    status: 'info',
  });
};

const failedToast = (parameter: string, message?: string) => {
  toast({
    id: 'PARAMETER_NOT_SET',
    title: t('toast.parameterNotSet'),
    description: message
      ? t('toast.parameterNotSetDescWithMessage', { parameter, message })
      : t('toast.parameterNotSetDesc', { parameter }),
    status: 'warning',
  });
};

const recallByHandler = async (arg: {
  metadata: unknown;
  handler: SingleMetadataHandler<any> | CollectionMetadataHandler<any[]>;
  store: AppStore;
  silent?: boolean;
}): Promise<boolean> => {
  const { metadata, handler, store, silent = false } = arg;

  let didRecall = false;

  try {
    const value = await handler.parse(metadata, store);
    handler.recall(value, store);
    didRecall = true;
  } catch {
    //
  }

  if (!silent) {
    if (didRecall) {
      successToast(t(handler.i18nKey));
    } else {
      failedToast(t(handler.i18nKey));
    }
  }

  return didRecall;
};

const recallByHandlers = async (arg: {
  metadata: unknown;
  handlers: (SingleMetadataHandler<any> | CollectionMetadataHandler<any[]>)[];
  store: AppStore;
  skip?: (SingleMetadataHandler<any> | CollectionMetadataHandler<any[]>)[];
  silent?: boolean;
}): Promise<Map<SingleMetadataHandler<any> | CollectionMetadataHandler<any[]>, unknown>> => {
  const { metadata, handlers, store, silent = false, skip = [] } = arg;

  const recalled = new Map<SingleMetadataHandler<any> | CollectionMetadataHandler<any[]>, unknown>();

  const filteredHandlers = handlers.filter(
    (handler) => !skip.some((skippedHandler) => skippedHandler.type === handler.type)
  );

  // It's possible for some metadata item's recall to clobber the recall of another. For example, the model recall
  // may change the width and height. If we are also recalling the width and height directly, we need to ensure that the
  // model is recalled first, so it doesn't accidentally override the width and height. This is the only known case
  // where the order of recall matters.
  const sortedHandlers = filteredHandlers.sort((a, b) => {
    if (a === ImageMetadataHandlers.MainModel) {
      return -1; // MainModel should be recalled first
    } else if (b === ImageMetadataHandlers.MainModel) {
      return 1; // MainModel should be recalled first
    } else {
      return 0; // Keep the original order for other handlers
    }
  });

  for (const handler of sortedHandlers) {
    try {
      const value = await handler.parse(metadata, store);
      handler.recall(value, store);
      recalled.set(handler, value);
    } catch {
      //
    }
  }

  if (!silent) {
    if (recalled.size > 0) {
      toast({
        id: 'PARAMETER_SET',
        title: t('toast.parametersSet'),
        status: 'info',
      });
    } else {
      toast({
        id: 'PARAMETER_SET',
        title: t('toast.parametersNotSet'),
        status: 'warning',
      });
    }
  }

  return recalled;
};

const recallImagePrompts = async (metadata: unknown, store: AppStore) => {
  const recalled = await recallByHandlers({
    metadata,
    handlers: [ImageMetadataHandlers.PositivePrompt, ImageMetadataHandlers.NegativePrompt],
    store,
    silent: true,
  });
  if (recalled.size > 0) {
    successToast(t('metadata.allPrompts'));
  }
};

const hasMetadataByHandlers = async (arg: {
  metadata: unknown;
  handlers: (SingleMetadataHandler<any> | CollectionMetadataHandler<any[]>)[];
  store: AppStore;
  require: 'some' | 'all';
}) => {
  const { metadata, handlers, store, require } = arg;
  for (const handler of handlers) {
    try {
      await handler.parse(metadata, store);
      if (require === 'some') {
        return true;
      }
    } catch {
      if (require === 'all') {
        return false;
      }
    }
  }
  return true;
};

const recallImageDimensions = async (metadata: unknown, store: AppStore) => {
  const recalled = await recallByHandlers({
    metadata,
    handlers: [ImageMetadataHandlers.Width, ImageMetadataHandlers.Height],
    store,
    silent: true,
  });
  if (recalled.size > 0) {
    successToast(t('metadata.imageDimensions'));
  }
};

const recallAllImageMetadata = async (
  metadata: unknown,
  store: AppStore,
  skip?: (SingleMetadataHandler<any> | CollectionMetadataHandler<any[]>)[]
) => {
  const handlers = Object.values(ImageMetadataHandlers).filter(
    (handler) => isSingleMetadataHandler(handler) || isCollectionMetadataHandler(handler)
  );
  await recallByHandlers({
    metadata,
    handlers,
    store,
    skip,
  });
};

export const MetadataUtils = {
  hasMetadataByHandlers,
  recallByHandler,
  recallByHandlers,
  recallAllImageMetadata,
  recallImagePrompts,
  recallImageDimensions,
} as const;

export function useSingleMetadataDatum<T>(metadata: unknown, handler: SingleMetadataHandler<T>) {
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
      try {
        const value = await handler.parse(metadata, store);
        setData(buildParsedSuccessData(value));
      } catch (error) {
        setData(buildParsedErrorData(WrappedError.wrap(error)));
      }
    },
    [handler, store]
  );

  useEffect(() => {
    parse(metadata);
  }, [metadata, parse]);

  const recall = useCallback(
    (value: T) => {
      handler.recall(value, store);
    },
    [handler, store]
  );

  return { data, recall };
}

export function useCollectionMetadataDatum<T extends any[]>(metadata: unknown, handler: CollectionMetadataHandler<T>) {
  const store = useAppStore();
  const [data, setData] = useState<Data<T>>(buildUnparsedData);

  const parse = useCallback(
    async (metadata: unknown) => {
      try {
        const value = await handler.parse(metadata, store);
        setData(buildParsedSuccessData(value));
      } catch (error) {
        setData(buildParsedErrorData(WrappedError.wrap(error)));
      }
    },
    [handler, store]
  );

  useEffect(() => {
    parse(metadata);
  }, [metadata, parse]);

  const recallAll = useCallback(
    (values: T) => {
      handler.recall(values, store);
    },
    [handler, store]
  );

  const recallOne = useCallback(
    (value: T[number]) => {
      handler.recallOne(value, store);
    },
    [handler, store]
  );

  return { data, recallAll, recallOne };
}

export function useUnrecallableMetadataDatum<T>(metadata: unknown, handler: UnrecallableMetadataHandler<T>) {
  const store = useAppStore();
  const [data, setData] = useState<Data<T>>(buildUnparsedData);

  const parse = useCallback(
    async (metadata: unknown) => {
      try {
        const value = await handler.parse(metadata, store);
        setData(buildParsedSuccessData(value));
      } catch (error) {
        setData(buildParsedErrorData(WrappedError.wrap(error)));
      }
    },
    [handler, store]
  );

  useEffect(() => {
    parse(metadata);
  }, [metadata, parse]);

  return { data };
}

const options = { subscribe: false };

const getModelIdentiferFromKey = async (key: string, store: AppStore): Promise<AnyModelConfig> => {
  const req = store.dispatch(modelsApi.endpoints.getModelConfig.initiate(key, options));
  const modelConfig = await req.unwrap();
  return modelConfig;
};

const parseModelIdentifier = async (raw: unknown, store: AppStore, type: ModelType): Promise<ModelIdentifierField> => {
  try {
    // First try the current format identifier: key, name, base, type, hash
    const { key } = zModelIdentifierField.parse(raw);
    const req = store.dispatch(modelsApi.endpoints.getModelConfig.initiate(key, options));
    const modelConfig = await req.unwrap();
    return zModelIdentifierField.parse(modelConfig);
  } catch {
    // We'll try to parse the old format identifier next
  }

  // Fall back to old format identifier: model_name, base_model
  // No error handling here - this is our last chance to get a model identifier
  const { model_name, base_model } = zModelIdentifier.parse(raw);
  const arg = { name: model_name, base: base_model, type };
  const req = store.dispatch(modelsApi.endpoints.getModelConfigByAttrs.initiate(arg, options));
  const modelConfig = await req.unwrap();
  return zModelIdentifierField.parse(modelConfig);
};

const isCompatibleWithMainModel = (candidate: ModelIdentifierField, store: AppStore) => {
  const base = selectBase(store.getState());
  if (!base) {
    return true;
  }
  return candidate.base === base;
};

const throwIfImageDoesNotExist = async (name: string, store: AppStore): Promise<void> => {
  try {
    await store.dispatch(imagesApi.endpoints.getImageDTO.initiate(name, { subscribe: false })).unwrap();
  } catch {
    throw new Error(`Image with name ${name} does not exist`);
  }
};

const throwIfModelDoesNotExist = async (key: string, store: AppStore): Promise<void> => {
  try {
    await store.dispatch(modelsApi.endpoints.getModelConfig.initiate(key, { subscribe: false }));
  } catch {
    throw new Error(`Model with key ${key} does not exist`);
  }
};
