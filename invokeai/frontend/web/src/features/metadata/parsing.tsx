/* eslint-disable @typescript-eslint/no-explicit-any */
import { Text } from '@invoke-ai/ui-library';
import type { AppStore } from 'app/store/store';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { WrappedError } from 'common/util/result';
import { get, isArray, isString } from 'es-toolkit/compat';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { bboxHeightChanged, bboxWidthChanged, canvasMetadataRecalled } from 'features/controlLayers/store/canvasSlice';
import { loraAllDeleted, loraRecalled } from 'features/controlLayers/store/lorasSlice';
import {
  animaQwen3EncoderModelSelected,
  animaVaeModelSelected,
  flux2DevMistralEncoderModelSelected,
  flux2VaeModelSelected,
  fluxVAESelected,
  geminiTemperatureChanged,
  geminiThinkingLevelChanged,
  heightChanged,
  imageSizeChanged,
  isValidKrea2RebalanceWeights,
  kleinQwen3EncoderModelSelected,
  krea2Qwen3VlEncoderModelSelected,
  krea2VaeModelSelected,
  minimaxH3DurationSecondsChanged,
  minimaxH3OutputModeChanged,
  minimaxH3TextEncoderModelSelected,
  minimaxH3TransformerModelSelected,
  negativePromptChanged,
  openaiBackgroundChanged,
  openaiInputFidelityChanged,
  openaiQualityChanged,
  positivePromptChanged,
  qwenImageComponentSourceSelected,
  qwenImageQuantizationChanged,
  qwenImageQwenVLEncoderModelSelected,
  qwenImageShiftChanged,
  qwenImageVaeModelSelected,
  refinerModelChanged,
  seedreamOptimizePromptChanged,
  seedreamWatermarkChanged,
  selectBase,
  setAnimaScheduler,
  setCfgRescaleMultiplier,
  setCfgScale,
  setClipSkip,
  setFluxDypeExponent,
  setFluxDypePreset,
  setFluxDypeScale,
  setFluxScheduler,
  setGuidance,
  setHiDiffusionEnabled,
  setHiDiffusionRauNetEnabled,
  setHiDiffusionT1Ratio,
  setHiDiffusionT2Ratio,
  setHiDiffusionWindowAttnEnabled,
  setIdeogram4ColorPalette,
  setIdeogram4GuidanceScale,
  setIdeogram4Mu,
  setIdeogram4SamplerPreset,
  setIdeogram4Steps,
  setImg2imgStrength,
  setKrea2RebalanceEnabled,
  setKrea2RebalanceMultiplier,
  setKrea2RebalanceWeights,
  setKrea2SeedVarianceEnabled,
  setKrea2SeedVarianceRandomizePercent,
  setKrea2SeedVarianceStrength,
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
  setZImageSeedVarianceEnabled,
  setZImageSeedVarianceRandomizePercent,
  setZImageSeedVarianceStrength,
  setZImageShift,
  t5EncoderModelSelected,
  vaeSelected,
  wanComponentSourceSelected,
  wanGuidanceScaleLowNoiseChanged,
  wanT5EncoderModelSelected,
  wanTransformerLowNoiseSelected,
  wanVaeModelSelected,
  widthChanged,
  zImageQwen3EncoderModelSelected,
  zImageQwen3SourceModelSelected,
  zImageVaeModelSelected,
} from 'features/controlLayers/store/paramsSlice';
import { refImagesRecalled } from 'features/controlLayers/store/refImagesSlice';
import type { CanvasMetadata, LoRA, RefImageState } from 'features/controlLayers/store/types';
import { zCanvasMetadata, zCanvasReferenceImageState_OLD, zRefImageState } from 'features/controlLayers/store/types';
import type { BaseModelType, ModelIdentifierField, ModelType } from 'features/nodes/types/common';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { zModelIdentifier } from 'features/nodes/types/v2/common';
import { modelSelected } from 'features/parameters/store/actions';
import type {
  ParameterCFGRescaleMultiplier,
  ParameterCFGScale,
  ParameterCLIPSkip,
  ParameterFluxDypeExponent,
  ParameterFluxDypePreset,
  ParameterFluxDypeScale,
  ParameterGuidance,
  ParameterHeight,
  ParameterIdeogram4SamplerPreset,
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
  zParameterFluxDypeExponent,
  zParameterFluxDypePreset,
  zParameterFluxDypeScale,
  zParameterGuidance,
  zParameterIdeogram4SamplerPreset,
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
import {
  isAnimaCompatibleVAEModelConfig,
  isAnimaQwen3EncoderModelConfig,
  isFlux1VAEModelConfig,
  isFlux2VAEModelConfig,
  isQwen3EncoderModelConfig,
} from 'services/api/types';
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

const getMetadataModelBase = (metadata: unknown): string | undefined => {
  const rawModel = getProperty(metadata, 'model');
  const modelBase = (rawModel as { base?: unknown } | undefined)?.base;
  return isString(modelBase) ? modelBase : undefined;
};

const assertMetadataModelBase = (metadata: unknown, expectedBase: string, handlerType: string): void => {
  const rawModel = getProperty(metadata, 'model');
  const modelBase = (rawModel as { base?: unknown } | undefined)?.base;
  assert(modelBase === expectedBase, `${handlerType} handler only works with ${expectedBase} metadata`);
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

export const parseMetadataHandler = <T,>(
  metadata: unknown,
  handler: { parse: (metadata: unknown, store: AppStore) => Promise<T> },
  store: AppStore
): Promise<T> => {
  return Promise.resolve().then(() => handler.parse(metadata, store));
};

const isSingleMetadataHandler = (
  handler: SingleMetadataHandler<any> | CollectionMetadataHandler<any[]> | UnrecallableMetadataHandler<any>
): handler is SingleMetadataHandler<any> => {
  return SingleMetadataKey in handler && handler[SingleMetadataKey] === true;
};

export const isCollectionMetadataHandler = (
  handler: SingleMetadataHandler<any> | CollectionMetadataHandler<any[]> | UnrecallableMetadataHandler<any>
): handler is CollectionMetadataHandler<any[]> => {
  return CollectionMetadataKey in handler && handler[CollectionMetadataKey] === true;
};

export const isUnrecallableMetadataHandler = (
  handler: SingleMetadataHandler<any> | CollectionMetadataHandler<any[]> | UnrecallableMetadataHandler<any>
): handler is UnrecallableMetadataHandler<any> => {
  return UnrecallableMetadataKey in handler && handler[UnrecallableMetadataKey] === true;
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
  parse: async (metadata, store) => {
    // guidance_embeds is inert for FLUX.2 Klein but genuinely consumed by FLUX.2 [dev]
    // (the graph sets guidance_embeds=True and passes the recorded guidance). So reject
    // only for non-dev FLUX.2: this displays and recalls the value for [dev] while never
    // leaking a stale value into the shared guidance param for Klein (shared with FLUX.1).
    // Resolve the image's own model to read its variant; if it can't be resolved (e.g.
    // uninstalled), fall back to skipping — same safe behavior as before for Klein.
    const rawModel = getProperty(metadata, 'model');
    const modelBase = (rawModel as { base?: unknown } | undefined)?.base;
    if (modelBase === 'flux2') {
      let isDev = false;
      try {
        const config = await resolveModel(
          rawModel as { key: string; hash?: string; name: string; base: string; type: string },
          store
        );
        isDev = 'variant' in config && config.variant === 'dev';
      } catch {
        isDev = false;
      }
      if (!isDev) {
        throw new Error('Guidance is not used for FLUX.2 Klein models.');
      }
    }
    const raw = getProperty(metadata, 'guidance');
    const parsed = zParameterGuidance.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setGuidance(value));
  },
  i18nKey: 'metadata.guidance',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterGuidance>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Guidance

//#region FluxDypePreset
const FluxDypePreset: SingleMetadataHandler<ParameterFluxDypePreset> = {
  [SingleMetadataKey]: true,
  type: 'FluxDypePreset',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'dype_preset');
    const parsed = zParameterFluxDypePreset.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setFluxDypePreset(value));
  },
  i18nKey: 'metadata.dypePreset',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterFluxDypePreset>) => (
    <MetadataPrimitiveValue value={value} />
  ),
};
//#endregion FluxDypePreset

//#region FluxDypeScale
const FluxDypeScale: SingleMetadataHandler<ParameterFluxDypeScale> = {
  [SingleMetadataKey]: true,
  type: 'FluxDypeScale',
  parse: (metadata, _store) => {
    // Only parse if preset is 'manual' (custom values)
    const preset = getProperty(metadata, 'dype_preset');
    if (preset !== 'manual') {
      throw new Error('DyPE scale only available when preset is "manual"');
    }
    const raw = getProperty(metadata, 'dype_scale');
    const parsed = zParameterFluxDypeScale.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setFluxDypeScale(value));
  },
  i18nKey: 'metadata.dypeScale',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterFluxDypeScale>) => (
    <MetadataPrimitiveValue value={value} />
  ),
};
//#endregion FluxDypeScale

//#region FluxDypeExponent
const FluxDypeExponent: SingleMetadataHandler<ParameterFluxDypeExponent> = {
  [SingleMetadataKey]: true,
  type: 'FluxDypeExponent',
  parse: (metadata, _store) => {
    // Only parse if preset is 'manual' (custom values)
    const preset = getProperty(metadata, 'dype_preset');
    if (preset !== 'manual') {
      throw new Error('DyPE exponent only available when preset is "manual"');
    }
    const raw = getProperty(metadata, 'dype_exponent');
    const parsed = zParameterFluxDypeExponent.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setFluxDypeExponent(value));
  },
  i18nKey: 'metadata.dypeExponent',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterFluxDypeExponent>) => (
    <MetadataPrimitiveValue value={value} />
  ),
};
//#endregion FluxDypeExponent

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
    if (base === 'flux' || base === 'flux2') {
      // Flux and Flux2 (Klein) only support euler, heun, lcm
      if (value === 'euler' || value === 'heun' || value === 'lcm') {
        store.dispatch(setFluxScheduler(value));
      }
    } else if (base === 'z-image') {
      // Z-Image supports euler, heun, lcm (but LCM only works well with Turbo, not Base)
      if (value === 'euler' || value === 'heun' || value === 'lcm') {
        store.dispatch(setZImageScheduler(value));
      }
    } else if (base === 'anima') {
      // Anima supports euler, heun, dpmpp_2m, dpmpp_2m_sde, er_sde, lcm
      if (
        value === 'euler' ||
        value === 'heun' ||
        value === 'dpmpp_2m' ||
        value === 'dpmpp_2m_sde' ||
        value === 'er_sde' ||
        value === 'lcm'
      ) {
        store.dispatch(setAnimaScheduler(value));
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

//#region HiDiffusion
const HiDiffusion: SingleMetadataHandler<boolean> = {
  [SingleMetadataKey]: true,
  type: 'HiDiffusion',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'hidiffusion');
    const parsed = raw === undefined ? false : z.boolean().parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setHiDiffusionEnabled(value));
  },
  i18nKey: 'metadata.hiDiffusion',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<boolean>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion HiDiffusion

//#region HiDiffusionRAUNet
const HiDiffusionRauNet: SingleMetadataHandler<boolean> = {
  [SingleMetadataKey]: true,
  type: 'HiDiffusionRauNet',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'hidiffusion_raunet');
    const parsed = z.boolean().parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setHiDiffusionRauNetEnabled(value));
  },
  i18nKey: 'metadata.hiDiffusionRauNet',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<boolean>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion HiDiffusionRAUNet

//#region HiDiffusionWindowAttn
const HiDiffusionWindowAttn: SingleMetadataHandler<boolean> = {
  [SingleMetadataKey]: true,
  type: 'HiDiffusionWindowAttn',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'hidiffusion_window_attn');
    const parsed = z.boolean().parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setHiDiffusionWindowAttnEnabled(value));
  },
  i18nKey: 'metadata.hiDiffusionWindowAttn',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<boolean>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion HiDiffusionWindowAttn

//#region HiDiffusionT1Ratio
const HiDiffusionT1Ratio: SingleMetadataHandler<number> = {
  [SingleMetadataKey]: true,
  type: 'HiDiffusionT1Ratio',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'hidiffusion_t1_ratio');
    const parsed = z.number().parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setHiDiffusionT1Ratio(value));
  },
  i18nKey: 'metadata.hiDiffusionT1Ratio',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<number>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion HiDiffusionT1Ratio

//#region HiDiffusionT2Ratio
const HiDiffusionT2Ratio: SingleMetadataHandler<number> = {
  [SingleMetadataKey]: true,
  type: 'HiDiffusionT2Ratio',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'hidiffusion_t2_ratio');
    const parsed = z.number().parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setHiDiffusionT2Ratio(value));
  },
  i18nKey: 'metadata.hiDiffusionT2Ratio',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<number>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion HiDiffusionT2Ratio

//#region ZImageSeedVarianceEnabled
const ZImageSeedVarianceEnabled: SingleMetadataHandler<boolean> = {
  [SingleMetadataKey]: true,
  type: 'ZImageSeedVarianceEnabled',
  parse: (metadata, _store) => {
    try {
      const raw = getProperty(metadata, 'z_image_seed_variance_enabled');
      const parsed = z.boolean().parse(raw);
      return Promise.resolve(parsed);
    } catch {
      // Default to false when metadata doesn't contain this field (e.g. older images)
      return Promise.resolve(false);
    }
  },
  recall: (value, store) => {
    store.dispatch(setZImageSeedVarianceEnabled(value));
  },
  i18nKey: 'metadata.seedVarianceEnabled',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<boolean>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion ZImageSeedVarianceEnabled

//#region ZImageSeedVarianceStrength
const ZImageSeedVarianceStrength: SingleMetadataHandler<number> = {
  [SingleMetadataKey]: true,
  type: 'ZImageSeedVarianceStrength',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'z_image_seed_variance_strength');
    const parsed = z.number().min(0).max(2).parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setZImageSeedVarianceStrength(value));
  },
  i18nKey: 'metadata.seedVarianceStrength',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<number>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion ZImageSeedVarianceStrength

//#region ZImageSeedVarianceRandomizePercent
const ZImageSeedVarianceRandomizePercent: SingleMetadataHandler<number> = {
  [SingleMetadataKey]: true,
  type: 'ZImageSeedVarianceRandomizePercent',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'z_image_seed_variance_randomize_percent');
    const parsed = z.number().min(1).max(100).parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setZImageSeedVarianceRandomizePercent(value));
  },
  i18nKey: 'metadata.seedVarianceRandomizePercent',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<number>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion ZImageSeedVarianceRandomizePercent

//#region QwenImageComponentSource
const QwenImageComponentSource: SingleMetadataHandler<ModelIdentifierField | null> = {
  [SingleMetadataKey]: true,
  type: 'QwenImageComponentSource',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'qwen_image_component_source');
    // Reject when the key is absent so the handler is not rendered for non-Qwen images
    if (raw === undefined) {
      return Promise.reject();
    }
    if (raw === null) {
      return Promise.resolve(null);
    }
    return Promise.resolve(zModelIdentifierField.parse(raw));
  },
  recall: (value, store) => {
    store.dispatch(qwenImageComponentSourceSelected(value));
  },
  i18nKey: 'modelManager.qwenImageComponentSource',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ModelIdentifierField | null>) => (
    <MetadataPrimitiveValue value={value ? value.name : 'None'} />
  ),
};
//#endregion QwenImageComponentSource

//#region QwenImageVaeModel
const QwenImageVaeModel: SingleMetadataHandler<ModelIdentifierField | null> = {
  [SingleMetadataKey]: true,
  type: 'QwenImageVaeModel',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'qwen_image_vae');
    // Reject when the key is absent so the handler is not rendered for non-Qwen images
    if (raw === undefined) {
      return Promise.reject();
    }
    if (raw === null) {
      return Promise.resolve(null);
    }
    return Promise.resolve(zModelIdentifierField.parse(raw));
  },
  recall: (value, store) => {
    store.dispatch(qwenImageVaeModelSelected(value));
  },
  i18nKey: 'modelManager.qwenImageVae',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ModelIdentifierField | null>) => (
    <MetadataPrimitiveValue value={value ? value.name : 'None'} />
  ),
};
//#endregion QwenImageVaeModel

//#region QwenImageQwenVLEncoderModel
const QwenImageQwenVLEncoderModel: SingleMetadataHandler<ModelIdentifierField | null> = {
  [SingleMetadataKey]: true,
  type: 'QwenImageQwenVLEncoderModel',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'qwen_image_qwen_vl_encoder');
    // Reject when the key is absent so the handler is not rendered for non-Qwen images
    if (raw === undefined) {
      return Promise.reject();
    }
    if (raw === null) {
      return Promise.resolve(null);
    }
    return Promise.resolve(zModelIdentifierField.parse(raw));
  },
  recall: (value, store) => {
    store.dispatch(qwenImageQwenVLEncoderModelSelected(value));
  },
  i18nKey: 'modelManager.qwenImageQwenVLEncoder',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ModelIdentifierField | null>) => (
    <MetadataPrimitiveValue value={value ? value.name : 'None'} />
  ),
};
//#endregion QwenImageQwenVLEncoderModel

//#region QwenImageQuantization
const QwenImageQuantization: SingleMetadataHandler<'none' | 'int8' | 'nf4'> = {
  [SingleMetadataKey]: true,
  type: 'QwenImageQuantization',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'qwen_image_quantization');
    // Reject when the key is absent so the handler is not rendered for non-Qwen images
    if (raw === undefined) {
      return Promise.reject();
    }
    const parsed = z.enum(['none', 'int8', 'nf4']).parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(qwenImageQuantizationChanged(value));
  },
  i18nKey: 'modelManager.qwenImageQuantization',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<'none' | 'int8' | 'nf4'>) => (
    <MetadataPrimitiveValue value={value} />
  ),
};
//#endregion QwenImageQuantization

//#region QwenImageShift
const QwenImageShift: SingleMetadataHandler<number | null> = {
  [SingleMetadataKey]: true,
  type: 'QwenImageShift',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'qwen_image_shift');
    // Reject when the key is absent so the handler is not rendered for non-Qwen images
    if (raw === undefined) {
      return Promise.reject();
    }
    if (raw === null) {
      return Promise.resolve(null);
    }
    const parsed = z.number().parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(qwenImageShiftChanged(value));
  },
  i18nKey: 'modelManager.qwenImageShift',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<number | null>) => (
    <MetadataPrimitiveValue value={value ?? 'Default'} />
  ),
};
//#endregion QwenImageShift

//#region WanTransformerLowNoise
const WanTransformerLowNoise: SingleMetadataHandler<ModelIdentifierField | null> = {
  [SingleMetadataKey]: true,
  type: 'WanTransformerLowNoise',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'wan_transformer_low_noise');
    // Reject when the key is absent so the handler is not rendered for non-Wan images
    if (raw === undefined) {
      return Promise.reject();
    }
    if (raw === null) {
      return Promise.resolve(null);
    }
    return Promise.resolve(zModelIdentifierField.parse(raw));
  },
  recall: (value, store) => {
    store.dispatch(wanTransformerLowNoiseSelected(value));
  },
  i18nKey: 'modelManager.wanTransformerLowNoise',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ModelIdentifierField | null>) => (
    <MetadataPrimitiveValue value={value ? value.name : 'None'} />
  ),
};
//#endregion WanTransformerLowNoise

//#region WanComponentSource
const WanComponentSource: SingleMetadataHandler<ModelIdentifierField | null> = {
  [SingleMetadataKey]: true,
  type: 'WanComponentSource',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'wan_component_source');
    if (raw === undefined) {
      return Promise.reject();
    }
    if (raw === null) {
      return Promise.resolve(null);
    }
    return Promise.resolve(zModelIdentifierField.parse(raw));
  },
  recall: (value, store) => {
    store.dispatch(wanComponentSourceSelected(value));
  },
  i18nKey: 'modelManager.wanComponentSource',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ModelIdentifierField | null>) => (
    <MetadataPrimitiveValue value={value ? value.name : 'None'} />
  ),
};
//#endregion WanComponentSource

//#region WanVaeModel
const WanVaeModel: SingleMetadataHandler<ModelIdentifierField | null> = {
  [SingleMetadataKey]: true,
  type: 'WanVaeModel',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'wan_vae_model');
    if (raw === undefined) {
      return Promise.reject();
    }
    if (raw === null) {
      return Promise.resolve(null);
    }
    return Promise.resolve(zModelIdentifierField.parse(raw));
  },
  recall: (value, store) => {
    store.dispatch(wanVaeModelSelected(value));
  },
  i18nKey: 'modelManager.wanVae',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ModelIdentifierField | null>) => (
    <MetadataPrimitiveValue value={value ? value.name : 'None'} />
  ),
};
//#endregion WanVaeModel

//#region WanT5EncoderModel
const WanT5EncoderModel: SingleMetadataHandler<ModelIdentifierField | null> = {
  [SingleMetadataKey]: true,
  type: 'WanT5EncoderModel',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'wan_t5_encoder_model');
    if (raw === undefined) {
      return Promise.reject();
    }
    if (raw === null) {
      return Promise.resolve(null);
    }
    return Promise.resolve(zModelIdentifierField.parse(raw));
  },
  recall: (value, store) => {
    store.dispatch(wanT5EncoderModelSelected(value));
  },
  i18nKey: 'modelManager.wanT5Encoder',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ModelIdentifierField | null>) => (
    <MetadataPrimitiveValue value={value ? value.name : 'None'} />
  ),
};
//#endregion WanT5EncoderModel

//#region WanGuidanceScaleLowNoise
const WanGuidanceScaleLowNoise: SingleMetadataHandler<number | null> = {
  [SingleMetadataKey]: true,
  type: 'WanGuidanceScaleLowNoise',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'wan_guidance_scale_low_noise');
    if (raw === undefined) {
      return Promise.reject();
    }
    if (raw === null) {
      return Promise.resolve(null);
    }
    const parsed = z.number().parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(wanGuidanceScaleLowNoiseChanged(value));
  },
  i18nKey: 'parameters.wanGuidanceScaleLowNoise',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<number | null>) => (
    <MetadataPrimitiveValue value={value ?? 'Default'} />
  ),
};
//#endregion WanGuidanceScaleLowNoise

//#region MiniMaxH3DurationSeconds
const MiniMaxH3DurationSeconds: SingleMetadataHandler<number> = {
  [SingleMetadataKey]: true,
  type: 'MiniMaxH3DurationSeconds',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'minimax_h3_duration_seconds');
    if (raw === undefined) {
      // Reject when the key is absent so the handler is not rendered for non-H3 media.
      return Promise.reject();
    }
    const parsed = z.number().int().min(5).max(14).parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(minimaxH3DurationSecondsChanged(value));
  },
  i18nKey: 'parameters.minimaxH3DurationSeconds',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<number>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion MiniMaxH3DurationSeconds

//#region MiniMaxH3OutputMode
const MiniMaxH3OutputMode: SingleMetadataHandler<'video' | 'image'> = {
  [SingleMetadataKey]: true,
  type: 'MiniMaxH3OutputMode',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'minimax_h3_output_mode');
    if (raw === undefined) {
      return Promise.reject();
    }
    const parsed = z.enum(['video', 'image']).parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(minimaxH3OutputModeChanged(value));
  },
  i18nKey: 'parameters.minimaxH3OutputMode',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<'video' | 'image'>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion MiniMaxH3OutputMode

//#region MiniMaxH3TransformerModel
const MiniMaxH3TransformerModel: SingleMetadataHandler<ModelIdentifierField | null> = {
  [SingleMetadataKey]: true,
  type: 'MiniMaxH3TransformerModel',
  parse: async (metadata, store) => {
    const raw = getProperty(metadata, 'minimax_h3_transformer_model');
    if (raw === undefined) {
      // The graph builder only writes this key when a single-file transformer override was
      // used; reject when absent so the handler is not rendered (and recall-all skips it).
      return Promise.reject();
    }
    if (raw === null) {
      return Promise.resolve(null);
    }
    // Validate the single-file transformer is still installed - recall-all must skip silently
    // (not clobber or error) when it has since been deleted.
    const parsed = await parseModelIdentifier(raw, store, 'main');
    assert(parsed.type === 'main' && parsed.base === 'minimax-h3');
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(minimaxH3TransformerModelSelected(value));
  },
  i18nKey: 'modelManager.minimaxH3TransformerModel',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ModelIdentifierField | null>) => (
    <MetadataPrimitiveValue value={value ? value.name : 'None'} />
  ),
};
//#endregion MiniMaxH3TransformerModel

//#region MiniMaxH3TextEncoderModel
const MiniMaxH3TextEncoderModel: SingleMetadataHandler<ModelIdentifierField | null> = {
  [SingleMetadataKey]: true,
  type: 'MiniMaxH3TextEncoderModel',
  parse: async (metadata, store) => {
    const raw = getProperty(metadata, 'minimax_h3_text_encoder_model');
    if (raw === undefined) {
      // The graph builder only writes this key when a single-file text-encoder override was
      // used; reject when absent so the handler is not rendered (and recall-all skips it).
      return Promise.reject();
    }
    if (raw === null) {
      return Promise.resolve(null);
    }
    // Validate the single-file encoder is still installed - recall-all must skip silently
    // (not clobber or error) when it has since been deleted.
    const parsed = await parseModelIdentifier(raw, store, 'qwen3_vl_encoder');
    assert(parsed.type === 'qwen3_vl_encoder' && parsed.base === 'minimax-h3');
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(minimaxH3TextEncoderModelSelected(value));
  },
  i18nKey: 'modelManager.minimaxH3TextEncoderModel',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ModelIdentifierField | null>) => (
    <MetadataPrimitiveValue value={value ? value.name : 'None'} />
  ),
};
//#endregion MiniMaxH3TextEncoderModel

//#region ZImageShift
const ZImageShift: SingleMetadataHandler<number | null> = {
  [SingleMetadataKey]: true,
  type: 'ZImageShift',
  parse: (metadata, store) => {
    const raw = getProperty(metadata, 'z_image_shift');
    if (raw === undefined) {
      // Older Z-Image images and new images generated with auto shift don't include this key.
      // Recall as null (auto) only when the recalled image is a Z-Image, so we don't clobber
      // the user's current shift when recalling unrelated metadata.
      const base = selectBase(store.getState());
      if (base !== 'z-image') {
        return Promise.reject();
      }
      return Promise.resolve(null);
    }
    // null or the 'auto' sentinel (written by the graph builder when shift is auto) recall as auto.
    if (raw === null || raw === 'auto') {
      return Promise.resolve(null);
    }
    const parsed = z.number().min(0).max(10).parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(setZImageShift(value));
  },
  i18nKey: 'metadata.zImageShift',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<number | null>) => {
    const { t } = useTranslation();
    return <MetadataPrimitiveValue value={value ?? t('common.auto')} />;
  },
};
//#endregion ZImageShift

//#region Ideogram4SamplerPreset
const Ideogram4SamplerPreset: SingleMetadataHandler<ParameterIdeogram4SamplerPreset> = {
  [SingleMetadataKey]: true,
  type: 'Ideogram4SamplerPreset',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'ideogram4_sampler_preset');
    const parsed = zParameterIdeogram4SamplerPreset.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    // Only recall onto an Ideogram 4 model so we don't set this (otherwise hidden) field for other bases.
    if (selectBase(store.getState()) !== 'ideogram-4') {
      return;
    }
    store.dispatch(setIdeogram4SamplerPreset(value));
  },
  i18nKey: 'parameters.samplerPreset',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterIdeogram4SamplerPreset>) => (
    <MetadataPrimitiveValue value={value} />
  ),
};
//#endregion Ideogram4SamplerPreset

//#region Ideogram4Steps
// Optional override of the preset step count. The graph writes 'auto' (sentinel) when unset; recall
// maps that back to null (= use preset). Only recalled onto an Ideogram 4 model.
const Ideogram4Steps: SingleMetadataHandler<number | null> = {
  [SingleMetadataKey]: true,
  type: 'Ideogram4Steps',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'ideogram4_steps');
    if (raw === undefined) {
      return Promise.reject();
    }
    if (raw === null || raw === 'auto') {
      return Promise.resolve(null);
    }
    // Backend requires steps >= 2; refuse a stale/out-of-range recalled value instead of recalling it.
    return Promise.resolve(z.number().int().min(2).max(100).parse(raw));
  },
  recall: (value, store) => {
    if (selectBase(store.getState()) !== 'ideogram-4') {
      return;
    }
    store.dispatch(setIdeogram4Steps(value));
  },
  i18nKey: 'parameters.steps',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<number | null>) => {
    const { t } = useTranslation();
    return <MetadataPrimitiveValue value={value ?? t('common.auto')} />;
  },
};
//#endregion Ideogram4Steps

//#region Ideogram4GuidanceScale
const Ideogram4GuidanceScale: SingleMetadataHandler<number | null> = {
  [SingleMetadataKey]: true,
  type: 'Ideogram4GuidanceScale',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'ideogram4_guidance_scale');
    if (raw === undefined) {
      return Promise.reject();
    }
    if (raw === null || raw === 'auto') {
      return Promise.resolve(null);
    }
    return Promise.resolve(z.number().min(1).max(20).parse(raw));
  },
  recall: (value, store) => {
    if (selectBase(store.getState()) !== 'ideogram-4') {
      return;
    }
    store.dispatch(setIdeogram4GuidanceScale(value));
  },
  i18nKey: 'parameters.guidance',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<number | null>) => {
    const { t } = useTranslation();
    return <MetadataPrimitiveValue value={value ?? t('common.auto')} />;
  },
};
//#endregion Ideogram4GuidanceScale

//#region Ideogram4Mu
const Ideogram4Mu: SingleMetadataHandler<number | null> = {
  [SingleMetadataKey]: true,
  type: 'Ideogram4Mu',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'ideogram4_mu');
    if (raw === undefined) {
      return Promise.reject();
    }
    if (raw === null || raw === 'auto') {
      return Promise.resolve(null);
    }
    return Promise.resolve(z.number().min(-4).max(4).parse(raw));
  },
  recall: (value, store) => {
    if (selectBase(store.getState()) !== 'ideogram-4') {
      return;
    }
    store.dispatch(setIdeogram4Mu(value));
  },
  i18nKey: 'parameters.shift',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<number | null>) => {
    const { t } = useTranslation();
    return <MetadataPrimitiveValue value={value ?? t('common.auto')} />;
  },
};
//#endregion Ideogram4Mu

//#region Ideogram4ColorPalette
const Ideogram4ColorPalette: SingleMetadataHandler<string[]> = {
  [SingleMetadataKey]: true,
  type: 'Ideogram4ColorPalette',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'ideogram4_color_palette');
    if (raw === undefined) {
      return Promise.reject();
    }
    return Promise.resolve(z.array(z.string()).parse(raw));
  },
  recall: (value, store) => {
    if (selectBase(store.getState()) !== 'ideogram-4') {
      return;
    }
    store.dispatch(setIdeogram4ColorPalette(value));
  },
  i18nKey: 'parameters.colorPalette',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<string[]>) => (
    <MetadataPrimitiveValue value={value.join(', ')} />
  ),
};
//#endregion Ideogram4ColorPalette

//#region Ideogram4Caption
// For regional/structured prompts the value actually encoded by the model is this assembled JSON
// caption, while `positive_prompt` holds the raw overall description (via the graph's decoy node).
// Recalling it into the positive prompt round-trips: the graph builder detects a leading `{` and passes
// the JSON through unchanged.
const Ideogram4Caption: SingleMetadataHandler<string> = {
  [SingleMetadataKey]: true,
  type: 'Ideogram4Caption',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'ideogram4_caption');
    if (raw === undefined) {
      return Promise.reject();
    }
    return Promise.resolve(z.string().parse(raw));
  },
  recall: (value, store) => {
    if (selectBase(store.getState()) !== 'ideogram-4') {
      return;
    }
    store.dispatch(positivePromptChanged(value));
  },
  i18nKey: 'parameters.ideogram4Caption',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<string>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Ideogram4Caption

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

/**
 * Bases with their own VAE slot in paramsSlice plus a dedicated handler, which nonetheless write to the
 * shared `metadata.vae` field. The generic VAEModel handler must not fire for them, else the metadata
 * panel renders a duplicate VAE row and "recall all" additionally writes into the (for those bases dead)
 * `params.vae` slot.
 *
 * When adding a base here: a dedicated handler dispatching into the correct slot MUST exist, and it MUST
 * be listed in IMAGE_METADATA_ACTION_HANDLERS - otherwise the row disappears without replacement.
 *
 * qwen-image and wan are deliberately absent: they write `qwen_image_vae` / `wan_vae_model` and never
 * collide with this handler in the first place.
 */
const BASES_WITH_DEDICATED_VAE_HANDLER: ReadonlySet<BaseModelType> = new Set([
  'flux', // Flux1VAEModel  -> params.fluxVAE
  'z-image', // ZImageVAEModel -> params.zImageVaeModel
  'flux2', // Flux2VAEModel  -> params.flux2VaeModel (Klein + [dev])
  'krea-2', // Krea2VAEModel  -> params.krea2VaeModel
  'anima', // AnimaVAEModel  -> params.animaVaeModel
]);

//#region VAEModel
const VAEModel: SingleMetadataHandler<ParameterVAEModel> = {
  [SingleMetadataKey]: true,
  type: 'VAEModel',
  parse: async (metadata, store) => {
    const raw = getProperty(metadata, 'vae');
    const parsed = await parseModelIdentifier(raw, store, 'vae');
    assert(parsed.type === 'vae');
    assert(isCompatibleWithMainModel(parsed, store));
    // Two axes, because either one alone leaves a hole. The selected base decides which slot is live;
    // the image's own base decides which handler owns the row. Without the provenance half, an Anima
    // image viewed with no main model selected (`base` null, startup or after the last model is
    // uninstalled) falls through to here and offers a recall into the - for Anima dead - `params.vae`
    // (review 4998711432).
    const metadataBase = getMetadataModelBase(metadata);
    assert(
      !metadataBase || !BASES_WITH_DEDICATED_VAE_HANDLER.has(metadataBase as BaseModelType),
      `VAEModel handler does not apply to "${metadataBase}" images - that base has a dedicated VAE handler`
    );
    const base = selectBase(store.getState());
    assert(
      !base || !BASES_WITH_DEDICATED_VAE_HANDLER.has(base),
      `VAEModel handler does not apply to base "${base}" - it has a dedicated VAE handler`
    );
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

//#region Flux1VAEModel
/**
 * FLUX.1 keeps its VAE in a dedicated slot (`params.fluxVAE`, read by buildFLUXGraph) but records it in
 * the shared `metadata.vae` field. Without this handler the generic VAEModel would recall it into
 * `params.vae`, which no FLUX graph ever reads - the recall looked like it worked but had no effect.
 */
const Flux1VAEModel: SingleMetadataHandler<ModelIdentifierField> = {
  [SingleMetadataKey]: true,
  type: 'Flux1VAEModel',
  parse: async (metadata, store) => {
    // Check provenance: `vae` is a shared field. Z-Image (and any base whose VAE pool includes FLUX
    // VAEs) can record a FLUX VAE, which would otherwise land in `params.fluxVAE` (review 4966712044).
    assertMetadataModelBase(metadata, 'flux', 'Flux1VAEModel');
    // The slot's domain is exactly its picker's (`useFlux1VAEModels`): a FLUX.1 VAE, standalone or as a
    // main model's bundled `vae` submodel. FLUX.2 VAEs are excluded - `flux_model_loader` cannot load
    // one - and so is every other base.
    const parsed = await parseVAEModelIdentifier({
      raw: getProperty(metadata, 'vae'),
      store,
      isCompatible: isFlux1VAEModelConfig,
      handlerType: 'Flux1VAEModel',
    });
    const base = selectBase(store.getState());
    assert(base === 'flux', 'Flux1VAEModel handler only works with FLUX.1 models');
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(fluxVAESelected(value));
  },
  i18nKey: 'metadata.vae',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ModelIdentifierField>) => (
    <MetadataPrimitiveValue value={`${value.name} (${value.base.toUpperCase()})`} />
  ),
};
//#endregion Flux1VAEModel

//#region ZImageQwen3EncoderModel
const ZImageQwen3EncoderModel: SingleMetadataHandler<ModelIdentifierField> = {
  [SingleMetadataKey]: true,
  type: 'ZImageQwen3EncoderModel',
  parse: async (metadata, store) => {
    // Check provenance: `qwen3_encoder` is also written by Anima and FLUX.2 Klein, and this handler
    // clears `zImageQwen3SourceModel` on recall (review 4966712044).
    assertMetadataModelBase(metadata, 'z-image', 'ZImageQwen3EncoderModel');
    // The picker's domain (`useQwen3EncoderModels`): the 4B/8B encoders, i.e. everything except Anima's
    // 0.6B, whose 1024-wide embeddings Z-Image cannot consume. That split lives in `variant`, so the
    // full config is needed - the identifier alone cannot tell the two apart.
    const parsed = await parseModelIdentifierMatching({
      raw: getProperty(metadata, 'qwen3_encoder'),
      store,
      type: 'qwen3_encoder',
      isCompatible: isQwen3EncoderModelConfig,
      handlerType: 'ZImageQwen3EncoderModel',
    });
    // Klein and Z-Image encoders both satisfy isQwen3EncoderModelConfig, so the variant cannot separate
    // those two - the currently selected base does.
    const base = selectBase(store.getState());
    assert(base === 'z-image', 'ZImageQwen3EncoderModel handler only works with Z-Image models');
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
//#endregion ZImageQwen3EncoderModel

//#region T5EncoderModel
const T5EncoderModel: SingleMetadataHandler<ModelIdentifierField> = {
  [SingleMetadataKey]: true,
  type: 'T5EncoderModel',
  parse: async (metadata, store) => {
    const raw = getProperty(metadata, 't5_encoder');
    const parsed = await parseModelIdentifier(raw, store, 't5_encoder');
    assert(parsed.type === 't5_encoder');
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(t5EncoderModelSelected(value));
  },
  i18nKey: 'metadata.t5Encoder',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ModelIdentifierField>) => (
    <MetadataPrimitiveValue value={`${value.name} (${value.base.toUpperCase()})`} />
  ),
};
//#endregion T5EncoderModel

//#region ZImageVAEModel
const ZImageVAEModel: SingleMetadataHandler<ModelIdentifierField> = {
  [SingleMetadataKey]: true,
  type: 'ZImageVAEModel',
  parse: async (metadata, store) => {
    // Check provenance: `vae` is a shared field, and this handler additionally clears
    // `zImageQwen3SourceModel` - foreign metadata must never reach it (review 4966712044).
    assertMetadataModelBase(metadata, 'z-image', 'ZImageVAEModel');
    // Z-Image borrows the FLUX.1 VAE pool - its picker is `useFlux1VAEModels` and it explicitly cannot
    // use a FLUX.2 VAE - so this slot has the same domain as `params.fluxVAE`, submodels included.
    const parsed = await parseVAEModelIdentifier({
      raw: getProperty(metadata, 'vae'),
      store,
      isCompatible: isFlux1VAEModelConfig,
      handlerType: 'ZImageVAEModel',
    });
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
    // `qwen3_source` is Z-Image-only today, but this handler clears both other Z-Image slots on recall,
    // so it is gated on provenance like its siblings rather than on the field being unique.
    assertMetadataModelBase(metadata, 'z-image', 'ZImageQwen3SourceModel');
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

//#region Krea2VAEModel
const Krea2VAEModel: SingleMetadataHandler<ModelIdentifierField> = {
  [SingleMetadataKey]: true,
  type: 'Krea2VAEModel',
  parse: async (metadata, store) => {
    assertMetadataModelBase(metadata, 'krea-2', 'Krea2VAEModel');
    const raw = getProperty(metadata, 'vae');
    const parsed = await parseModelIdentifier(raw, store, 'vae');
    assert(parsed.type === 'vae');
    assert(parsed.base === 'qwen-image' || parsed.base === 'anima', 'Krea2VAEModel requires a Qwen Image or Anima VAE');
    // Only recall if the current main model is Krea-2 (its VAE dropdown differs from other bases).
    const base = selectBase(store.getState());
    assert(base === 'krea-2', 'Krea2VAEModel handler only works with Krea-2 models');
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(krea2VaeModelSelected(value));
  },
  i18nKey: 'metadata.vae',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ModelIdentifierField>) => (
    <MetadataPrimitiveValue value={`${value.name} (${value.base.toUpperCase()})`} />
  ),
};
//#endregion Krea2VAEModel

//#region Krea2Qwen3VlEncoderModel
const Krea2Qwen3VlEncoderModel: SingleMetadataHandler<ModelIdentifierField> = {
  [SingleMetadataKey]: true,
  type: 'Krea2Qwen3VlEncoderModel',
  parse: async (metadata, store) => {
    assertMetadataModelBase(metadata, 'krea-2', 'Krea2Qwen3VlEncoderModel');
    const raw = getProperty(metadata, 'qwen3_vl_encoder');
    const parsed = await parseModelIdentifier(raw, store, 'qwen3_vl_encoder');
    assert(parsed.type === 'qwen3_vl_encoder');
    const base = selectBase(store.getState());
    assert(base === 'krea-2', 'Krea2Qwen3VlEncoderModel handler only works with Krea-2 models');
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(krea2Qwen3VlEncoderModelSelected(value));
  },
  i18nKey: 'metadata.krea2Qwen3VlEncoder',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ModelIdentifierField>) => (
    <MetadataPrimitiveValue value={`${value.name} (${value.base.toUpperCase()})`} />
  ),
};
//#endregion Krea2Qwen3VlEncoderModel

//#region Krea2SeedVarianceEnabled
const Krea2SeedVarianceEnabled: SingleMetadataHandler<boolean> = {
  [SingleMetadataKey]: true,
  type: 'Krea2SeedVarianceEnabled',
  parse: (metadata, store) => {
    // Only applies to Krea-2 models, and only when the field is actually present — otherwise recalling
    // an unrelated/older image would silently clear the user's current enhancer state. (A synchronous
    // throw here is turned into a rejected promise by the parse runner, skipping the handler.)
    assert(selectBase(store.getState()) === 'krea-2', 'Krea2SeedVarianceEnabled handler only applies to Krea-2 models');
    assertMetadataModelBase(metadata, 'krea-2', 'Krea2SeedVarianceEnabled');
    const raw = getProperty(metadata, 'krea2_seed_variance_enabled');
    return Promise.resolve(z.boolean().parse(raw));
  },
  recall: (value, store) => {
    store.dispatch(setKrea2SeedVarianceEnabled(value));
  },
  i18nKey: 'metadata.seedVarianceEnabled',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<boolean>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Krea2SeedVarianceEnabled

//#region Krea2SeedVarianceStrength
const Krea2SeedVarianceStrength: SingleMetadataHandler<number> = {
  [SingleMetadataKey]: true,
  type: 'Krea2SeedVarianceStrength',
  parse: (metadata, store) => {
    assert(
      selectBase(store.getState()) === 'krea-2',
      'Krea2SeedVarianceStrength handler only applies to Krea-2 models'
    );
    assertMetadataModelBase(metadata, 'krea-2', 'Krea2SeedVarianceStrength');
    const raw = getProperty(metadata, 'krea2_seed_variance_strength');
    // Strength is a multiplier of the embedding std, capped at 2 (matches the invocation + param state).
    return Promise.resolve(z.number().min(0).max(2).parse(raw));
  },
  recall: (value, store) => {
    store.dispatch(setKrea2SeedVarianceStrength(value));
  },
  i18nKey: 'metadata.seedVarianceStrength',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<number>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Krea2SeedVarianceStrength

//#region Krea2SeedVarianceRandomizePercent
const Krea2SeedVarianceRandomizePercent: SingleMetadataHandler<number> = {
  [SingleMetadataKey]: true,
  type: 'Krea2SeedVarianceRandomizePercent',
  parse: (metadata, store) => {
    assert(
      selectBase(store.getState()) === 'krea-2',
      'Krea2SeedVarianceRandomizePercent handler only applies to Krea-2 models'
    );
    assertMetadataModelBase(metadata, 'krea-2', 'Krea2SeedVarianceRandomizePercent');
    const raw = getProperty(metadata, 'krea2_seed_variance_randomize_percent');
    // 0 is the valid "disabled" value (matches the slider, param state, and invocation); reject negatives.
    return Promise.resolve(z.number().min(0).max(100).parse(raw));
  },
  recall: (value, store) => {
    store.dispatch(setKrea2SeedVarianceRandomizePercent(value));
  },
  i18nKey: 'metadata.seedVarianceRandomizePercent',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<number>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Krea2SeedVarianceRandomizePercent

//#region Krea2RebalanceEnabled
const Krea2RebalanceEnabled: SingleMetadataHandler<boolean> = {
  [SingleMetadataKey]: true,
  type: 'Krea2RebalanceEnabled',
  parse: (metadata, store) => {
    assert(selectBase(store.getState()) === 'krea-2', 'Krea2RebalanceEnabled handler only applies to Krea-2 models');
    assertMetadataModelBase(metadata, 'krea-2', 'Krea2RebalanceEnabled');
    const raw = getProperty(metadata, 'krea2_rebalance_enabled');
    return Promise.resolve(z.boolean().parse(raw));
  },
  recall: (value, store) => {
    store.dispatch(setKrea2RebalanceEnabled(value));
  },
  i18nKey: 'metadata.krea2RebalanceEnabled',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<boolean>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Krea2RebalanceEnabled

//#region Krea2RebalanceMultiplier
const Krea2RebalanceMultiplier: SingleMetadataHandler<number> = {
  [SingleMetadataKey]: true,
  type: 'Krea2RebalanceMultiplier',
  parse: (metadata, store) => {
    assert(selectBase(store.getState()) === 'krea-2', 'Krea2RebalanceMultiplier handler only applies to Krea-2 models');
    assertMetadataModelBase(metadata, 'krea-2', 'Krea2RebalanceMultiplier');
    const raw = getProperty(metadata, 'krea2_rebalance_multiplier');
    return Promise.resolve(z.number().min(0).max(20).parse(raw));
  },
  recall: (value, store) => {
    store.dispatch(setKrea2RebalanceMultiplier(value));
  },
  i18nKey: 'metadata.krea2RebalanceMultiplier',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<number>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Krea2RebalanceMultiplier

//#region Krea2RebalanceWeights
const Krea2RebalanceWeights: SingleMetadataHandler<string> = {
  [SingleMetadataKey]: true,
  type: 'Krea2RebalanceWeights',
  parse: (metadata, store) => {
    assert(selectBase(store.getState()) === 'krea-2', 'Krea2RebalanceWeights handler only applies to Krea-2 models');
    assertMetadataModelBase(metadata, 'krea-2', 'Krea2RebalanceWeights');
    const raw = getProperty(metadata, 'krea2_rebalance_weights');
    // Only recall a string the backend rebalance node would actually accept (exactly 12 finite numbers),
    // so recalling stale/garbage metadata can't dispatch state that later fails at generation time.
    return Promise.resolve(
      z.string().refine(isValidKrea2RebalanceWeights, 'expected exactly 12 finite comma-separated numbers').parse(raw)
    );
  },
  recall: (value, store) => {
    store.dispatch(setKrea2RebalanceWeights(value));
  },
  i18nKey: 'metadata.krea2RebalanceWeights',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<string>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Krea2RebalanceWeights

//#region AnimaVAEModel
const AnimaVAEModel: SingleMetadataHandler<ModelIdentifierField> = {
  [SingleMetadataKey]: true,
  type: 'AnimaVAEModel',
  parse: async (metadata, store) => {
    // Check provenance: `vae` is a shared field (SD/SDXL/SD3/FLUX/FLUX.2/Krea-2/Z-Image write it too).
    // Without this, a Krea-2 image recalled while Anima is selected would push its Qwen-Image VAE into
    // the Anima slot.
    assertMetadataModelBase(metadata, 'anima', 'AnimaVAEModel');
    // Defers to the Anima VAE picker's own domain (isAnimaCompatibleVAEModelConfig): an Anima-base
    // (Wan/QwenImage) VAE, a FLUX VAE, or a 16-channel Wan VAE - all of which anima_l2i / anima_i2l
    // accept. Gating on the full config matters here beyond the submodel question: the Wan arm reads
    // `latent_channels`, which the identifier does not carry (reviews 4966712044, 4972570279).
    const parsed = await parseVAEModelIdentifier({
      raw: getProperty(metadata, 'vae'),
      store,
      isCompatible: isAnimaCompatibleVAEModelConfig,
      handlerType: 'AnimaVAEModel',
    });
    const base = selectBase(store.getState());
    assert(base === 'anima', 'AnimaVAEModel handler only works with Anima models');
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(animaVaeModelSelected(value));
  },
  i18nKey: 'metadata.vae',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ModelIdentifierField>) => (
    <MetadataPrimitiveValue value={`${value.name} (${value.base.toUpperCase()})`} />
  ),
};
//#endregion AnimaVAEModel

//#region AnimaQwen3EncoderModel
const AnimaQwen3EncoderModel: SingleMetadataHandler<ModelIdentifierField> = {
  [SingleMetadataKey]: true,
  type: 'AnimaQwen3EncoderModel',
  parse: async (metadata, store) => {
    // Check provenance: `qwen3_encoder` is also written by Z-Image and FLUX.2 Klein.
    assertMetadataModelBase(metadata, 'anima', 'AnimaQwen3EncoderModel');
    // Deliberately no `base` assert - Anima encoders are identified by `variant`, not by base - but the
    // variant itself must be checked, which needs the full config: Anima's text encoder produces
    // 1024-wide embeddings, and a 4B (2560) or 8B (4096) encoder recalled into this slot fails the next
    // generation. This is the picker's own domain (`useAnimaQwen3EncoderModels`).
    const parsed = await parseModelIdentifierMatching({
      raw: getProperty(metadata, 'qwen3_encoder'),
      store,
      type: 'qwen3_encoder',
      isCompatible: isAnimaQwen3EncoderModelConfig,
      handlerType: 'AnimaQwen3EncoderModel',
    });
    const base = selectBase(store.getState());
    assert(base === 'anima', 'AnimaQwen3EncoderModel handler only works with Anima models');
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(animaQwen3EncoderModelSelected(value));
  },
  i18nKey: 'metadata.qwen3Encoder',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ModelIdentifierField>) => (
    <MetadataPrimitiveValue value={`${value.name} (${value.base.toUpperCase()})`} />
  ),
};
//#endregion AnimaQwen3EncoderModel

//#region Flux2VAEModel
/**
 * FLUX.2 Klein and FLUX.2 [dev] share a single VAE slot (`flux2VaeModel`) and the same
 * `metadata.vae` field — both draw from the 32-channel AutoencoderKLFlux2 pool — so one
 * handler covers both variants and no dev/Klein disambiguation is needed on recall.
 */
const Flux2VAEModel: SingleMetadataHandler<ModelIdentifierField> = {
  [SingleMetadataKey]: true,
  type: 'Flux2VAEModel',
  parse: async (metadata, store) => {
    // Check provenance: `vae` is a shared field - e.g. a Krea-2 image (Qwen-Image VAE) recalled while
    // FLUX.2 is selected would otherwise land in `params.flux2VaeModel`.
    assertMetadataModelBase(metadata, 'flux2', 'Flux2VAEModel');
    // Same domain as the picker (`useFlux2VAEModels`), which both Klein and [dev] share.
    const parsed = await parseVAEModelIdentifier({
      raw: getProperty(metadata, 'vae'),
      store,
      isCompatible: isFlux2VAEModelConfig,
      handlerType: 'Flux2VAEModel',
    });
    const base = selectBase(store.getState());
    assert(base === 'flux2', 'Flux2VAEModel handler only works with FLUX.2 models');
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(flux2VaeModelSelected(value));
  },
  i18nKey: 'metadata.vae',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ModelIdentifierField>) => (
    <MetadataPrimitiveValue value={`${value.name} (${value.base.toUpperCase()})`} />
  ),
};
//#endregion Flux2VAEModel

//#region KleinQwen3EncoderModel
const KleinQwen3EncoderModel: SingleMetadataHandler<ModelIdentifierField> = {
  [SingleMetadataKey]: true,
  type: 'KleinQwen3EncoderModel',
  parse: async (metadata, store) => {
    // `qwen3_encoder` is no longer Klein-only: Z-Image and Anima write it too (into their own slots),
    // so provenance decides, not just the field being present. FLUX.2 [dev] never writes it, so one
    // flux2 check covers both variants.
    assertMetadataModelBase(metadata, 'flux2', 'KleinQwen3EncoderModel');
    // Same domain as the picker (`useQwen3EncoderModels`) - Klein 4B/9B, never Anima's 0.6B.
    const parsed = await parseModelIdentifierMatching({
      raw: getProperty(metadata, 'qwen3_encoder'),
      store,
      type: 'qwen3_encoder',
      isCompatible: isQwen3EncoderModelConfig,
      handlerType: 'KleinQwen3EncoderModel',
    });
    const base = selectBase(store.getState());
    assert(base === 'flux2', 'KleinQwen3EncoderModel handler only works with FLUX.2 Klein models');
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(kleinQwen3EncoderModelSelected(value));
  },
  i18nKey: 'metadata.qwen3Encoder',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ModelIdentifierField>) => (
    <MetadataPrimitiveValue value={`${value.name} (${value.base.toUpperCase()})`} />
  ),
};
//#endregion KleinQwen3EncoderModel

//#region Flux2DevMistralEncoderModel
const Flux2DevMistralEncoderModel: SingleMetadataHandler<ModelIdentifierField> = {
  [SingleMetadataKey]: true,
  type: 'Flux2DevMistralEncoderModel',
  parse: async (metadata, store) => {
    const raw = getProperty(metadata, 'mistral_encoder');
    const parsed = await parseModelIdentifier(raw, store, 'mistral_encoder');
    assert(parsed.type === 'mistral_encoder');
    // mistral_encoder is dev-only metadata; Klein never writes it. Just gate on
    // base. (parseModelIdentifier already rejects when the field is absent.)
    const base = selectBase(store.getState());
    assert(base === 'flux2', 'Flux2DevMistralEncoderModel handler only works with FLUX.2 models');
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(flux2DevMistralEncoderModelSelected(value));
  },
  i18nKey: 'metadata.mistralEncoder',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<ModelIdentifierField>) => (
    <MetadataPrimitiveValue value={`${value.name} (${value.base.toUpperCase()})`} />
  ),
};
//#endregion Flux2DevMistralEncoderModel

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
          const modelConfig = await getModelIdentiferFromKey(key, store);
          identifier = zModelIdentifierField.parse(modelConfig);
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
        const resolvedConfig = await resolveModel(entity.controlAdapter.model, store);
        entity.controlAdapter.model = zModelIdentifierField.parse(resolvedConfig);
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
          const resolvedConfig = await resolveModel(refImage.config.model, store);
          refImage.config.model = zModelIdentifierField.parse(resolvedConfig);
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
      // FLUX.2 reference images don't have a model field (built-in support)
      if ('model' in refImage.config && refImage.config.model) {
        const resolvedConfig = await resolveModel(refImage.config.model, store);
        // Update the model reference in case the key changed (e.g. model was reinstalled)
        refImage.config.model = zModelIdentifierField.parse(resolvedConfig);
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
    // FLUX.2 reference images don't have a model field (built-in support)
    if ('model' in value.config && value.config.model) {
      return <MetadataPrimitiveValue value={value.config.model.name} />;
    }
    return <MetadataPrimitiveValue value="No model" />;
  },
};
//#endregion RefImages

//#region External Image Size
const ImageSize: SingleMetadataHandler<string> = {
  [SingleMetadataKey]: true,
  type: 'ImageSize',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'image_size');
    const parsed = z.string().min(1).parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(imageSizeChanged(value));
  },
  i18nKey: 'metadata.imageSize',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<string>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion External Image Size

//#region Gemini Temperature
const GeminiTemperature: SingleMetadataHandler<number> = {
  [SingleMetadataKey]: true,
  type: 'GeminiTemperature',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'gemini_temperature');
    const parsed = z.number().min(0).max(2).parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(geminiTemperatureChanged(value));
  },
  i18nKey: 'metadata.geminiTemperature',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<number>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Gemini Temperature

//#region Gemini Thinking Level
const zGeminiThinkingLevel = z.enum(['minimal', 'high']);
const GeminiThinkingLevel: SingleMetadataHandler<'minimal' | 'high'> = {
  [SingleMetadataKey]: true,
  type: 'GeminiThinkingLevel',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'gemini_thinking_level');
    const parsed = zGeminiThinkingLevel.parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(geminiThinkingLevelChanged(value));
  },
  i18nKey: 'metadata.geminiThinkingLevel',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<'minimal' | 'high'>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Gemini Thinking Level

//#region OpenAI Quality
const OpenaiQuality: SingleMetadataHandler<'auto' | 'high' | 'medium' | 'low'> = {
  [SingleMetadataKey]: true,
  type: 'OpenaiQuality',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'openai_quality');
    const parsed = z.enum(['auto', 'high', 'medium', 'low']).parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(openaiQualityChanged(value));
  },
  i18nKey: 'metadata.openaiQuality',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<'auto' | 'high' | 'medium' | 'low'>) => (
    <MetadataPrimitiveValue value={value} />
  ),
};
//#endregion OpenAI Quality

//#region OpenAI Background
const OpenaiBackground: SingleMetadataHandler<'auto' | 'transparent' | 'opaque'> = {
  [SingleMetadataKey]: true,
  type: 'OpenaiBackground',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'openai_background');
    const parsed = z.enum(['auto', 'transparent', 'opaque']).parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(openaiBackgroundChanged(value));
  },
  i18nKey: 'metadata.openaiBackground',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<'auto' | 'transparent' | 'opaque'>) => (
    <MetadataPrimitiveValue value={value} />
  ),
};
//#endregion OpenAI Background

//#region OpenAI Input Fidelity
const OpenaiInputFidelity: SingleMetadataHandler<'low' | 'high'> = {
  [SingleMetadataKey]: true,
  type: 'OpenaiInputFidelity',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'openai_input_fidelity');
    const parsed = z.enum(['low', 'high']).parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(openaiInputFidelityChanged(value));
  },
  i18nKey: 'metadata.openaiInputFidelity',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<'low' | 'high'>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion OpenAI Input Fidelity

//#region Seedream Watermark
const SeedreamWatermark: SingleMetadataHandler<boolean> = {
  [SingleMetadataKey]: true,
  type: 'SeedreamWatermark',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'seedream_watermark');
    const parsed = z.boolean().parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(seedreamWatermarkChanged(value));
  },
  i18nKey: 'metadata.seedreamWatermark',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<boolean>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Seedream Watermark

//#region Seedream Optimize Prompt
const SeedreamOptimizePrompt: SingleMetadataHandler<boolean> = {
  [SingleMetadataKey]: true,
  type: 'SeedreamOptimizePrompt',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'seedream_optimize_prompt');
    const parsed = z.boolean().parse(raw);
    return Promise.resolve(parsed);
  },
  recall: (value, store) => {
    store.dispatch(seedreamOptimizePromptChanged(value));
  },
  i18nKey: 'metadata.seedreamOptimizePrompt',
  LabelComponent: MetadataLabel,
  ValueComponent: ({ value }: SingleMetadataValueProps<boolean>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Seedream Optimize Prompt

export const ImageMetadataHandlers = {
  CreatedBy,
  GenerationMode,
  PositivePrompt,
  NegativePrompt,
  CFGScale,
  CFGRescaleMultiplier,
  CLIPSkip,
  Guidance,
  FluxDypePreset,
  FluxDypeScale,
  FluxDypeExponent,
  Width,
  Height,
  Seed,
  Steps,
  DenoisingStrength,
  SeamlessX,
  SeamlessY,
  HiDiffusion,
  HiDiffusionRauNet,
  HiDiffusionWindowAttn,
  HiDiffusionT1Ratio,
  HiDiffusionT2Ratio,
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
  Flux1VAEModel,
  ZImageQwen3EncoderModel,
  T5EncoderModel,
  ZImageVAEModel,
  ZImageQwen3SourceModel,
  AnimaVAEModel,
  AnimaQwen3EncoderModel,
  Flux2VAEModel,
  KleinQwen3EncoderModel,
  Flux2DevMistralEncoderModel,
  ZImageSeedVarianceEnabled,
  ZImageSeedVarianceStrength,
  ZImageSeedVarianceRandomizePercent,
  Krea2VAEModel,
  Krea2Qwen3VlEncoderModel,
  Krea2SeedVarianceEnabled,
  Krea2SeedVarianceStrength,
  Krea2SeedVarianceRandomizePercent,
  Krea2RebalanceEnabled,
  Krea2RebalanceMultiplier,
  Krea2RebalanceWeights,
  QwenImageComponentSource,
  QwenImageVaeModel,
  QwenImageQwenVLEncoderModel,
  QwenImageQuantization,
  QwenImageShift,
  WanTransformerLowNoise,
  WanComponentSource,
  WanVaeModel,
  WanT5EncoderModel,
  WanGuidanceScaleLowNoise,
  MiniMaxH3DurationSeconds,
  MiniMaxH3OutputMode,
  MiniMaxH3TransformerModel,
  MiniMaxH3TextEncoderModel,
  ZImageShift,
  Ideogram4SamplerPreset,
  Ideogram4Steps,
  Ideogram4GuidanceScale,
  Ideogram4Mu,
  Ideogram4ColorPalette,
  Ideogram4Caption,
  LoRAs,
  CanvasLayers,
  RefImages,
  ImageSize,
  GeminiTemperature,
  GeminiThinkingLevel,
  OpenaiQuality,
  OpenaiBackground,
  OpenaiInputFidelity,
  SeedreamWatermark,
  SeedreamOptimizePrompt,
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
    const value = await parseMetadataHandler(metadata, handler, store);
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
      const value = await parseMetadataHandler(metadata, handler, store);
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
      await parseMetadataHandler(metadata, handler, store);
      if (require === 'some') {
        return true;
      }
    } catch {
      if (require === 'all') {
        return false;
      }
    }
  }
  return require === 'all';
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

/**
 * The selected base, subscribed to so a change re-runs the parse.
 *
 * Handlers read `selectBase(store.getState())` imperatively inside `parse` - they have to, because parsing
 * also happens outside React (recall-all, hotkeys). A row therefore carries the verdict of whichever base
 * was selected when it mounted, and the metadata viewer stays open across model switches: without this
 * subscription, rows that should now appear stay hidden and rows that should now be hidden keep a live
 * recall button pointed at a slot that is no longer in play.
 */
const useSelectedBaseForReparse = (): ReturnType<typeof selectBase> => useAppSelector(selectBase);

/**
 * Runs a recall only if the handler's own gate still admits the metadata.
 *
 * The reparse driven by `useSelectedBaseForReparse` is asynchronous, so there is a window in which a row is
 * still on screen under a base that no longer admits it. Re-running the gate here closes that window: a
 * recall that is no longer valid does nothing rather than writing a foreign model into an inactive slot.
 *
 * The passed value is what gets recalled, not the reparsed one - for a collection the row owns a single
 * item, and the row must do what it displays.
 *
 * @returns whether the recall ran.
 */
export const recallIfStillValid = async <TValue,>(arg: {
  metadata: unknown;
  handler: { parse: (metadata: unknown, store: AppStore) => Promise<unknown> };
  recall: (value: TValue, store: AppStore) => void;
  value: TValue;
  store: AppStore;
}): Promise<boolean> => {
  const { metadata, handler, recall, value, store } = arg;
  try {
    await parseMetadataHandler(metadata, handler, store);
  } catch {
    // No longer applicable under the current base - the pending reparse is about to drop the row.
    return false;
  }
  recall(value, store);
  return true;
};

const useRevalidatedRecall = <TValue,>(
  metadata: unknown,
  handler: { parse: (metadata: unknown, store: AppStore) => Promise<unknown>; i18nKey: string },
  recall: (value: TValue, store: AppStore) => void
) => {
  const store = useAppStore();

  return useCallback(
    (value: TValue) => {
      void recallIfStillValid({ metadata, handler, recall, value, store }).then((didRecall) => {
        if (!didRecall) {
          // The button is still on screen because the reparse has not landed yet. Silently doing
          // nothing reads as a broken button, so say so - same toast `recallByHandler` uses when a
          // parse fails.
          failedToast(t(handler.i18nKey));
        }
      });
    },
    [metadata, handler, store, recall]
  );
};

/**
 * The parse verdict for one metadata row, against the store's *current* state.
 *
 * Extracted from the hooks below so it can be tested directly: the verdict is a function of the selected
 * base as much as of the metadata (handlers read `selectBase(store.getState())` imperatively inside
 * `parse`), and there is no DOM test framework here to drive a hook through a base switch.
 */
export const parseMetadataDatum = async <T,>(
  metadata: unknown,
  handler: { parse: (metadata: unknown, store: AppStore) => Promise<T> },
  store: AppStore
): Promise<ParsedSuccessData<T> | ParsedErrorData> => {
  try {
    return buildParsedSuccessData(await parseMetadataHandler(metadata, handler, store));
  } catch (error) {
    return buildParsedErrorData(WrappedError.wrap(error));
  }
};

/**
 * Keeps one row's parse verdict in sync with the metadata, the handler and the selected base.
 *
 * Single source for all three datum hooks: the `base` dependency is the whole point of the subscription
 * (see `useSelectedBaseForReparse`), and having it written out three times invited exactly one of them
 * to be forgotten.
 */
const useMetadataDatum = <T,>(
  metadata: unknown,
  handler: { parse: (metadata: unknown, store: AppStore) => Promise<T> }
): Data<T> => {
  const store = useAppStore();
  const base = useSelectedBaseForReparse();
  const [data, setData] = useState<Data<T>>(buildUnparsedData);

  useEffect(() => {
    let isActive = true;

    void parseMetadataDatum(metadata, handler, store).then((next) => {
      if (isActive) {
        setData(next);
      }
    });

    return () => {
      isActive = false;
    };
  }, [metadata, handler, store, base]);

  return data;
};

export function useSingleMetadataDatum<T>(metadata: unknown, handler: SingleMetadataHandler<T>) {
  const data = useMetadataDatum(metadata, handler);
  const recall = useRevalidatedRecall(metadata, handler, handler.recall);

  return { data, recall };
}

export function useCollectionMetadataDatum<T extends any[]>(metadata: unknown, handler: CollectionMetadataHandler<T>) {
  const data = useMetadataDatum(metadata, handler);
  const recallAll = useRevalidatedRecall(metadata, handler, handler.recall);
  const recallOne = useRevalidatedRecall(metadata, handler, handler.recallOne);

  return { data, recallAll, recallOne };
}

export function useUnrecallableMetadataDatum<T>(metadata: unknown, handler: UnrecallableMetadataHandler<T>) {
  const data = useMetadataDatum(metadata, handler);

  return { data };
}

const options = { subscribe: false };

const getModelIdentiferFromKey = async (key: string, store: AppStore): Promise<AnyModelConfig> => {
  const req = store.dispatch(modelsApi.endpoints.getModelConfig.initiate(key, options));
  const modelConfig = await req.unwrap();
  return modelConfig;
};

/**
 * Resolve a metadata model reference to the installed model's *full* config. Handlers that must gate
 * on more than base and type - e.g. a VAE's latent geometry - need the whole config; the identifier
 * fields that actually get recalled into state are what `parseModelIdentifier` narrows this down to.
 */
const parseModelConfig = async (raw: unknown, store: AppStore, type: ModelType): Promise<AnyModelConfig> => {
  try {
    // First try the current format identifier: key, name, base, type, hash
    const { key } = zModelIdentifierField.parse(raw);
    const req = store.dispatch(modelsApi.endpoints.getModelConfig.initiate(key, options));
    const modelConfig = await req.unwrap();
    // Discarded on purpose - the config is what we return; this only asserts it is identifiable, so a
    // config that isn't still falls through to the lookups below.
    zModelIdentifierField.parse(modelConfig);
    return modelConfig;
  } catch {
    // We'll try hash-based lookup next
  }

  // Try hash-based lookup (handles reinstalled models with new UUID keys)
  try {
    const { hash } = zModelIdentifierField.parse(raw);
    if (hash) {
      const req = store.dispatch(modelsApi.endpoints.getModelConfigByHash.initiate(hash, options));
      const modelConfig = await req.unwrap();
      zModelIdentifierField.parse(modelConfig);
      return modelConfig;
    }
  } catch {
    // We'll try the old format identifier next
  }

  // Fall back to old format identifier: model_name, base_model
  // No error handling here - this is our last chance to get a model identifier
  const { model_name, base_model } = zModelIdentifier.parse(raw);
  const arg = { name: model_name, base: base_model, type };
  const req = store.dispatch(modelsApi.endpoints.getModelConfigByAttrs.initiate(arg, options));
  const modelConfig = await req.unwrap();
  zModelIdentifierField.parse(modelConfig);
  return modelConfig;
};

const parseModelIdentifier = async (raw: unknown, store: AppStore, type: ModelType): Promise<ModelIdentifierField> => {
  return zModelIdentifierField.parse(await parseModelConfig(raw, store, type));
};

/**
 * Resolve a metadata model reference and gate it on the *full config*, not on the identifier.
 *
 * A `ModelIdentifierField` carries only key/hash/name/base/type, so a handler asserting on it accepts
 * every sibling of the right type. Where a slot's domain is defined by `variant` - Qwen3 encoders split
 * into Anima's 0.6B (hidden_size 1024) and the 4B/8B ones Z-Image and Klein use - that is not enough:
 * recalling the wrong variant fills the slot with an encoder whose embeddings the transformer cannot
 * consume (review 4997022178).
 *
 * @param isCompatible the slot's own domain, i.e. the same guard its picker is built from.
 * @param handlerType used in the assertion message, so a rejected row is traceable to its handler.
 */
const parseModelIdentifierMatching = async (arg: {
  raw: unknown;
  store: AppStore;
  type: ModelType;
  isCompatible: (config: AnyModelConfig) => boolean;
  handlerType: string;
}): Promise<ModelIdentifierField> => {
  const { raw, store, type, isCompatible, handlerType } = arg;
  const config = await parseModelConfig(raw, store, type);
  assert(isCompatible(config), `${handlerType} requires a model this slot can hold`);
  return zModelIdentifierField.parse(config);
};

/**
 * Resolve a `vae` metadata reference to the identifier a VAE slot should be filled with.
 *
 * A VAE slot may legitimately hold a *main model's bundled `vae` submodel*: every VAE picker is built
 * from a `is*VAEModelConfig` guard called without `excludeSubmodels`, so main models with a `vae`
 * submodel show up as options, and the graph builders then record that identifier verbatim. Asserting
 * `parsed.type === 'vae'` - as each of these handlers used to - therefore dropped the metadata row
 * outright for exactly the models the picker offered.
 *
 * Gating on the full config rather than on the identifier also lets callers check properties that the
 * identifier does not carry, such as a Wan VAE's `latent_channels`.
 *
 * @param isCompatible the slot's own domain, i.e. the same guard its picker is built from.
 * @param handlerType used in the assertion messages, so a rejected row is traceable to its handler.
 */
const parseVAEModelIdentifier = async (arg: {
  raw: unknown;
  store: AppStore;
  isCompatible: (config: AnyModelConfig, excludeSubmodels?: boolean) => boolean;
  handlerType: string;
}): Promise<ModelIdentifierField> => {
  const { raw, store, isCompatible, handlerType } = arg;
  const rawIdentifier = zModelIdentifierField.safeParse(raw).data;
  const config = await parseModelConfig(raw, store, 'vae');
  // A main model only qualifies when the reference actually points at its VAE. An absent submodel_type
  // counts: the linear-UI graph builders record the picker's identifier as-is, without one.
  const isMainVaeSubmodel =
    config.type === 'main' &&
    rawIdentifier?.type === 'main' &&
    (rawIdentifier.submodel_type === null ||
      rawIdentifier.submodel_type === undefined ||
      rawIdentifier.submodel_type === 'vae');
  assert(
    config.type !== 'main' || isMainVaeSubmodel,
    `${handlerType} requires a VAE model or a main model VAE submodel`
  );
  assert(isCompatible(config, !isMainVaeSubmodel), `${handlerType} requires a VAE this slot can hold`);
  // Normalised so the recalled value is unambiguous downstream - a bare main-model identifier in a VAE
  // slot would be indistinguishable from a main-model selection.
  return zModelIdentifierField.parse({
    ...config,
    ...(isMainVaeSubmodel ? { submodel_type: 'vae' } : {}),
  });
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

/**
 * Resolve a model by key, falling back to hash or name+base+type lookup if the key is not found.
 * This handles the case where a model was deleted and reinstalled (getting a new UUID key).
 * Fallback order: key → hash → name+base+type
 * Returns the resolved model config, or throws if the model cannot be found by any method.
 */
const resolveModel = async (
  model: { key: string; hash?: string; name: string; base: string; type: string },
  store: AppStore
): Promise<AnyModelConfig> => {
  // First try by key (fast path)
  try {
    const req = store.dispatch(modelsApi.endpoints.getModelConfig.initiate(model.key, { subscribe: false }));
    return await req.unwrap();
  } catch {
    // Key not found - try fallback
  }

  // Second try by hash (most reliable for reinstalled models - hash is content-based)
  if (model.hash) {
    try {
      const req = store.dispatch(modelsApi.endpoints.getModelConfigByHash.initiate(model.hash, { subscribe: false }));
      return await req.unwrap();
    } catch {
      // Hash not found - try next fallback
    }
  }

  // Last resort: look up by name + base + type
  try {
    const req = store.dispatch(
      modelsApi.endpoints.getModelConfigByAttrs.initiate(
        { name: model.name, base: model.base as any, type: model.type as any },
        { subscribe: false }
      )
    );
    return await req.unwrap();
  } catch {
    throw new Error(`Model "${model.name}" (key: ${model.key}) does not exist`);
  }
};
