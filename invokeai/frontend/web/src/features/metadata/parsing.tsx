/* eslint-disable @typescript-eslint/no-explicit-any */
import { Text } from '@invoke-ai/ui-library';
import type { AppStore } from 'app/store/store';
import { useAppStore } from 'app/store/storeHooks';
import { withResultAsync } from 'common/util/result';
import { get, isArray, isString } from 'es-toolkit/compat';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { bboxHeightChanged, bboxWidthChanged } from 'features/controlLayers/store/canvasSlice';
import { loraAllDeleted, loraRecalled } from 'features/controlLayers/store/lorasSlice';
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
import type { LoRA } from 'features/controlLayers/store/types';
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
  zLoRAWeight,
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
import type { ComponentType } from 'react';
import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { modelsApi } from 'services/api/endpoints/models';
import type { AnyModelConfig, ModelType } from 'services/api/types';
import { assert } from 'tsafe';
import z from 'zod/v4';

const MetadataLabel = ({ i18nKey }: { i18nKey: string }) => {
  const { t } = useTranslation();
  return (
    <Text as="span" fontWeight="semibold" whiteSpace="pre-wrap" me={2}>
      {t(i18nKey)}:
    </Text>
  );
};

const MetadataLabelWithCount = <T extends any[]>({ i18nKey, i }: { i18nKey: string; i: number; values: T }) => {
  const { t } = useTranslation();
  return (
    <Text as="span" fontWeight="semibold" whiteSpace="pre-wrap" me={2}>
      {`${t(i18nKey)} ${i + 1}:`}
    </Text>
  );
};

const MetadataPrimitiveValue = ({ value }: { value: string | number | boolean | null | undefined }) => {
  return <Text as="span">{value}</Text>;
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

type ParsedSuccessData<T> = {
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

export type Data<T> = UnparsedData | ParsedSuccessData<T> | ParsedErrorData;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
type SingleMetadataLabelProps<T> = {
  value: T;
};
type SingleMetadataValueProps<T> = {
  value: T;
};
export type SingleMetadataHandler<T> = {
  type: string;
  parse: (metadata: unknown, store: AppStore) => Promise<T> | T;
  recall: (value: T, store: AppStore) => void;
  LabelComponent: ComponentType<SingleMetadataLabelProps<T>>;
  ValueComponent: ComponentType<SingleMetadataValueProps<T>>;
};

type CollectionMetadataLabelProps<T extends any[]> = {
  values: T;
  i: number;
};
type CollectionMetadataValueProps<T extends any[]> = {
  value: T[number];
};
export type CollectionMetadataHandler<T extends any[]> = {
  type: string;
  parse: (metadata: unknown, store: AppStore) => Promise<T> | T;
  recallAll: (values: T, store: AppStore) => void;
  recallItem: (value: T[number], store: AppStore) => void;
  LabelComponent: ComponentType<CollectionMetadataLabelProps<T>>;
  ValueComponent: ComponentType<CollectionMetadataValueProps<T>>;
};

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
type UnrecallableMetadataLabelProps<T> = {
  value: T;
};
type UnrecallableMetadataValueProps<T> = {
  value: T;
};
export type UnrecallableMetadataHandler<T> = {
  type: string;
  parse: (metadata: unknown, store: AppStore) => Promise<T> | T;
  LabelComponent: ComponentType<UnrecallableMetadataLabelProps<T>>;
  ValueComponent: ComponentType<UnrecallableMetadataValueProps<T>>;
};

//#region Created By
const CreatedBy: UnrecallableMetadataHandler<string> = {
  type: 'CreatedBy',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'created_by');
    const parsed = z.string().parse(raw);
    return parsed;
  },
  LabelComponent: () => <MetadataLabel i18nKey="metadata.createdBy" />,
  ValueComponent: ({ value }: UnrecallableMetadataLabelProps<string>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Created By

//#region Generation Mode
const GenerationMode: UnrecallableMetadataHandler<string> = {
  type: 'GenerationMode',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'generation_mode');
    const parsed = z.string().parse(raw);
    return parsed;
  },
  LabelComponent: () => <MetadataLabel i18nKey="metadata.generationMode" />,
  ValueComponent: ({ value }: UnrecallableMetadataLabelProps<string>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Generation Mode

//#region Positive Prompt
const PositivePrompt: SingleMetadataHandler<ParameterPositivePrompt> = {
  type: 'PositivePrompt',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'positive_prompt');
    const parsed = zParameterPositivePrompt.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(positivePromptChanged(value));
  },
  LabelComponent: () => <MetadataLabel i18nKey="metadata.positivePrompt" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterPositivePrompt>) => (
    <MetadataPrimitiveValue value={value} />
  ),
};
//#endregion Positive Prompt

//#region Negative Prompt
const NegativePrompt: SingleMetadataHandler<ParameterNegativePrompt> = {
  type: 'NegativePrompt',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'negative_prompt');
    const parsed = zParameterNegativePrompt.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(negativePromptChanged(value));
  },
  LabelComponent: () => <MetadataLabel i18nKey="metadata.negativePrompt" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterNegativePrompt>) => (
    <MetadataPrimitiveValue value={value} />
  ),
};
//#endregion Negative Prompt

//#region SDXL Positive Style Prompt
const PositiveStylePrompt: SingleMetadataHandler<ParameterPositiveStylePromptSDXL> = {
  type: 'PositiveStylePrompt',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'positive_style_prompt');
    const parsed = zParameterPositiveStylePromptSDXL.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(positivePrompt2Changed(value));
  },
  LabelComponent: () => <MetadataLabel i18nKey="sdxl.posStylePrompt" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterPositiveStylePromptSDXL>) => (
    <MetadataPrimitiveValue value={value} />
  ),
};
//#endregion SDXL Positive Style Prompt

//#region SDXL Negative Style Prompt
const NegativeStylePrompt: SingleMetadataHandler<ParameterPositiveStylePromptSDXL> = {
  type: 'NegativeStylePrompt',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'negative_style_prompt');
    const parsed = zParameterNegativeStylePromptSDXL.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(negativePrompt2Changed(value));
  },
  LabelComponent: () => <MetadataLabel i18nKey="sdxl.negStylePrompt" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterPositiveStylePromptSDXL>) => (
    <MetadataPrimitiveValue value={value} />
  ),
};
//#endregion SDXL Negative Style Prompt

//#region CFG Scale
const CFGScale: SingleMetadataHandler<ParameterCFGScale> = {
  type: 'CFGScale',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'cfg_scale');
    const parsed = zParameterCFGScale.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setCfgScale(value));
  },
  LabelComponent: () => <MetadataLabel i18nKey="metadata.cfgScale" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterCFGScale>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion CFG Scale

//#region CFG Rescale Multiplier
const CFGRescaleMultiplier: SingleMetadataHandler<ParameterCFGRescaleMultiplier> = {
  type: 'CFGRescaleMultiplier',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'cfg_rescale_multiplier');
    const parsed = zParameterCFGRescaleMultiplier.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setCfgRescaleMultiplier(value));
  },
  LabelComponent: () => <MetadataLabel i18nKey="metadata.cfgRescaleMultiplier" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterCFGRescaleMultiplier>) => (
    <MetadataPrimitiveValue value={value} />
  ),
};
//#endregion CFG Rescale Multiplier

//#region Guidance
const Guidance: SingleMetadataHandler<ParameterGuidance> = {
  type: 'Guidance',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'guidance');
    const parsed = zParameterGuidance.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setGuidance(value));
  },
  LabelComponent: () => <MetadataLabel i18nKey="metadata.guidance" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterGuidance>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Guidance

//#region Scheduler
const Scheduler: SingleMetadataHandler<ParameterScheduler> = {
  type: 'Scheduler',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'scheduler');
    const parsed = zParameterScheduler.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setScheduler(value));
  },
  LabelComponent: () => <MetadataLabel i18nKey="metadata.scheduler" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterScheduler>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Scheduler

//#region Width
const Width: SingleMetadataHandler<ParameterWidth> = {
  type: 'Width',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'width');
    const parsed = zParameterImageDimension.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(bboxWidthChanged({ width: value, updateAspectRatio: true, clamp: true }));
  },
  LabelComponent: () => <MetadataLabel i18nKey="metadata.width" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterWidth>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Width

//#region Height
const Height: SingleMetadataHandler<ParameterHeight> = {
  type: 'Height',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'height');
    const parsed = zParameterImageDimension.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(bboxHeightChanged({ height: value, updateAspectRatio: true, clamp: true }));
  },
  LabelComponent: () => <MetadataLabel i18nKey="metadata.height" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterHeight>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Height

//#region Seed
const Seed: SingleMetadataHandler<ParameterSeed> = {
  type: 'Seed',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'seed');
    const parsed = zParameterSeed.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setSeed(value));
  },
  LabelComponent: () => <MetadataLabel i18nKey="metadata.seed" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterSeed>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Seed

//#region Steps
const Steps: SingleMetadataHandler<ParameterSteps> = {
  type: 'Steps',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'steps');
    const parsed = zParameterSteps.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setSteps(value));
  },
  LabelComponent: () => <MetadataLabel i18nKey="metadata.steps" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterSteps>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion Steps

//#region DenoisingStrength
const DenoisingStrength: SingleMetadataHandler<ParameterStrength> = {
  type: 'DenoisingStrength',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'strength');
    const parsed = zParameterStrength.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setImg2imgStrength(value));
  },
  LabelComponent: () => <MetadataLabel i18nKey="metadata.strength" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterStrength>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion DenoisingStrength

//#region SeamlessX
const SeamlessX: SingleMetadataHandler<ParameterSeamlessX> = {
  type: 'SeamlessX',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'seamless_x');
    const parsed = zParameterSeamlessX.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setSeamlessXAxis(value));
  },
  LabelComponent: () => <MetadataLabel i18nKey="metadata.seamlessXAxis" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterSeamlessX>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion SeamlessX

//#region SeamlessY
const SeamlessY: SingleMetadataHandler<ParameterSeamlessY> = {
  type: 'SeamlessY',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'seamless_y');
    const parsed = zParameterSeamlessY.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setSeamlessYAxis(value));
  },
  LabelComponent: () => <MetadataLabel i18nKey="metadata.seamlessYAxis" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterSeamlessY>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion SeamlessY

//#region RefinerModel
const RefinerModel: SingleMetadataHandler<ParameterSDXLRefinerModel> = {
  type: 'RefinerModel',
  parse: async (metadata, store) => {
    const raw = getProperty(metadata, 'refiner_model');
    const parsed = await parseModelIdentifier(raw, store, 'main');
    assert(parsed.type === 'main');
    assert(parsed.base === 'sdxl-refiner');
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(refinerModelChanged(value));
  },
  LabelComponent: () => <MetadataLabel i18nKey="sdxl.refinermodel" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterSDXLRefinerModel>) => (
    <MetadataPrimitiveValue value={`${value.name} (${value.base.toUpperCase()})`} />
  ),
};
//#endregion RefinerModel

//#region RefinerSteps
const RefinerSteps: SingleMetadataHandler<ParameterSteps> = {
  type: 'RefinerSteps',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'refiner_steps');
    const parsed = zParameterSteps.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setRefinerSteps(value));
  },
  LabelComponent: () => <MetadataLabel i18nKey="sdxl.refinerSteps" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterSteps>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion RefinerSteps

//#region RefinerCFGScale
const RefinerCFGScale: SingleMetadataHandler<ParameterCFGScale> = {
  type: 'RefinerCFGScale',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'refiner_cfg_scale');
    const parsed = zParameterCFGScale.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setRefinerCFGScale(value));
  },
  LabelComponent: () => <MetadataLabel i18nKey="sdxl.cfgScale" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterCFGScale>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion RefinerCFGScale

//#region RefinerScheduler
const RefinerScheduler: SingleMetadataHandler<ParameterScheduler> = {
  type: 'RefinerScheduler',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'refiner_scheduler');
    const parsed = zParameterScheduler.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setRefinerScheduler(value));
  },
  LabelComponent: () => <MetadataLabel i18nKey="sdxl.scheduler" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterScheduler>) => <MetadataPrimitiveValue value={value} />,
};
//#endregion RefinerScheduler

//#region RefinerPositiveAestheticScore
const RefinerPositiveAestheticScore: SingleMetadataHandler<ParameterSDXLRefinerPositiveAestheticScore> = {
  type: 'RefinerPositiveAestheticScore',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'refiner_positive_aesthetic_score');
    const parsed = zParameterSDXLRefinerPositiveAestheticScore.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setRefinerPositiveAestheticScore(value));
  },
  LabelComponent: () => <MetadataLabel i18nKey="sdxl.posAestheticScore" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterSDXLRefinerPositiveAestheticScore>) => (
    <MetadataPrimitiveValue value={value} />
  ),
};
//#endregion RefinerPositiveAestheticScore

//#region RefinerNegativeAestheticScore
const RefinerNegativeAestheticScore: SingleMetadataHandler<ParameterSDXLRefinerNegativeAestheticScore> = {
  type: 'RefinerNegativeAestheticScore',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'refiner_negative_aesthetic_score');
    const parsed = zParameterSDXLRefinerNegativeAestheticScore.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setRefinerNegativeAestheticScore(value));
  },
  LabelComponent: () => <MetadataLabel i18nKey="sdxl.negAestheticScore" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterSDXLRefinerNegativeAestheticScore>) => (
    <MetadataPrimitiveValue value={value} />
  ),
};
//#endregion RefinerNegativeAestheticScore

//#region RefinerDenoisingStart
const RefinerDenoisingStart: SingleMetadataHandler<ParameterSDXLRefinerStart> = {
  type: 'RefinerDenoisingStart',
  parse: (metadata, _store) => {
    const raw = getProperty(metadata, 'refiner_start');
    const parsed = zParameterSDXLRefinerStart.parse(raw);
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(setRefinerStart(value));
  },
  LabelComponent: () => <MetadataLabel i18nKey="sdxl.refinerStart" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterSDXLRefinerStart>) => (
    <MetadataPrimitiveValue value={value} />
  ),
};
//#endregion RefinerDenoisingStart

//#region MainModel
const MainModel: SingleMetadataHandler<ParameterModel> = {
  type: 'MainModel',
  parse: async (metadata, store) => {
    const raw = getProperty(metadata, 'model');
    const parsed = await parseModelIdentifier(raw, store, 'main');
    assert(parsed.type === 'main');
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(modelSelected(value));
  },
  LabelComponent: () => <MetadataLabel i18nKey="metadata.model" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterModel>) => (
    <MetadataPrimitiveValue value={`${value.name} (${value.base.toUpperCase()})`} />
  ),
};
//#endregion MainModel

//#region VAEModel
const VAEModel: SingleMetadataHandler<ParameterVAEModel> = {
  type: 'VAEModel',
  parse: async (metadata, store) => {
    const raw = getProperty(metadata, 'vae');
    const parsed = await parseModelIdentifier(raw, store, 'vae');
    assert(parsed.type === 'vae');
    return parsed;
  },
  recall: (value, store) => {
    store.dispatch(vaeSelected(value));
  },
  LabelComponent: () => <MetadataLabel i18nKey="metadata.vae" />,
  ValueComponent: ({ value }: SingleMetadataValueProps<ParameterVAEModel>) => (
    <MetadataPrimitiveValue value={`${value.name} (${value.base.toUpperCase()})`} />
  ),
};
//#endregion VAEModel

//#region LoRAs
const LoRAs: CollectionMetadataHandler<LoRA[]> = {
  type: 'LoRAs',
  parse: async (metadata, store) => {
    const rawArray = getProperty(metadata, 'loras');
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
          // Old format - { lora : { key } }
          const key = getProperty(rawItem, 'lora.key');
          assert(isString(key));
          const modelConfig = await getModelConfig(key, store);
          identifier = zModelIdentifierField.parse(modelConfig);
        }
        assert(identifier.type === 'lora');
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
  recallItem: (value, store) => {
    store.dispatch(loraRecalled({ lora: value }));
  },
  recallAll: (values, store) => {
    store.dispatch(loraAllDeleted());
    for (const lora of values) {
      store.dispatch(loraRecalled({ lora }));
    }
  },
  LabelComponent: ({ values, i }: CollectionMetadataLabelProps<LoRA[]>) => (
    <MetadataLabelWithCount i18nKey="models.lora" values={values} i={i} />
  ),
  ValueComponent: ({ value }: CollectionMetadataValueProps<LoRA[]>) => (
    <MetadataPrimitiveValue value={`${value.model.name} (${value.model.base.toUpperCase()}) - ${value.weight}`} />
  ),
};
//#endregion LoRAs

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
  LoRAs,
} satisfies Record<
  string,
  UnrecallableMetadataHandler<any> | SingleMetadataHandler<any> | CollectionMetadataHandler<any[]>
>;

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
      const result = await withResultAsync(async () => await Promise.resolve(handler.parse(metadata, store)));
      if (result.isOk()) {
        setData(buildParsedSuccessData(result.value));
      } else {
        setData(buildParsedErrorData(result.error));
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

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
export function useCollectionMetadataDatum<T extends any[]>(metadata: unknown, handler: CollectionMetadataHandler<T>) {
  const store = useAppStore();
  const [data, setData] = useState<Data<T>>(buildUnparsedData);

  const parse = useCallback(
    async (metadata: unknown) => {
      const result = await withResultAsync(async () => await Promise.resolve(handler.parse(metadata, store)));
      if (result.isOk()) {
        setData(buildParsedSuccessData(result.value));
      } else {
        setData(buildParsedErrorData(result.error));
      }
    },
    [handler, store]
  );

  useEffect(() => {
    parse(metadata);
  }, [metadata, parse]);

  const recallAll = useCallback(() => {
    if (!data.isSuccess) {
      return;
    }
    handler.recallAll(data.value, store);
  }, [data.isSuccess, data.value, handler, store]);

  const recallItem = useCallback(
    (item: T) => {
      handler.recallItem(item, store);
    },
    [handler, store]
  );

  return { data, recallAll, recallItem };
}

export function useUnrecallableMetadataDatum<T>(metadata: unknown, handler: UnrecallableMetadataHandler<T>) {
  const store = useAppStore();
  const [data, setData] = useState<Data<T>>(buildUnparsedData);

  const parse = useCallback(
    async (metadata: unknown) => {
      const result = await withResultAsync(async () => await Promise.resolve(handler.parse(metadata, store)));
      if (result.isOk()) {
        setData(buildParsedSuccessData(result.value));
      } else {
        setData(buildParsedErrorData(result.error));
      }
    },
    [handler, store]
  );

  useEffect(() => {
    parse(metadata);
  }, [metadata, parse]);

  return { data };
}

const getModelConfig = async (key: string, store: AppStore): Promise<AnyModelConfig> => {
  const modelConfig = await store
    .dispatch(modelsApi.endpoints.getModelConfig.initiate(key, { subscribe: false }))
    .unwrap();
  return modelConfig;
};

const parseModelIdentifier = async (raw: unknown, store: AppStore, type: ModelType): Promise<ModelIdentifierField> => {
  // First try the current format identifier: key, name, base, type, hash
  try {
    const { key } = zModelIdentifierField.parse(raw);
    const req = store.dispatch(modelsApi.endpoints.getModelConfig.initiate(key, { subscribe: false }));
    const modelConfig = await req.unwrap();
    return zModelIdentifierField.parse(modelConfig);
  } catch {
    // noop
  }

  // Fall back to old format identifier: model_name, base_model
  try {
    const { model_name: name, base_model: base } = zModelIdentifier.parse(raw);
    const req = store.dispatch(
      modelsApi.endpoints.getModelConfigByAttrs.initiate({ name, base, type }, { subscribe: false })
    );
    const modelConfig = await req.unwrap();
    return zModelIdentifierField.parse(modelConfig);
  } catch {
    // noop
  }
  throw new Error('Unable to parse model identifier');
};
