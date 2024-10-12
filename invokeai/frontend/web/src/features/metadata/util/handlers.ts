import { getStore } from 'app/store/nanostores/store';
import { deepClone } from 'common/util/deepClone';
import { objectKeys } from 'common/util/objectKeys';
import { shouldConcatPromptsChanged } from 'features/controlLayers/store/paramsSlice';
import type { LoRA } from 'features/controlLayers/store/types';
import type {
  BuildMetadataHandlers,
  MetadataGetLabelFunc,
  MetadataHandlers,
  MetadataParseFunc,
  MetadataRecallFunc,
  MetadataRenderValueFunc,
  MetadataValidateFunc,
} from 'features/metadata/types';
import { fetchModelConfig } from 'features/metadata/util/modelFetchingHelpers';
import { validators } from 'features/metadata/util/validators';
import type { ModelIdentifierField } from 'features/nodes/types/common';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { size } from 'lodash-es';

import { parsers } from './parsers';
import { recallers } from './recallers';

const renderModelConfigValue: MetadataRenderValueFunc<ModelIdentifierField> = async (value) => {
  try {
    const modelConfig = await fetchModelConfig(value.key);
    return `${modelConfig.name} (${modelConfig.base.toUpperCase()})`;
  } catch {
    return `${value.key} (${value.base.toUpperCase()})`;
  }
};
const renderLoRAValue: MetadataRenderValueFunc<LoRA> = async (value) => {
  try {
    const modelConfig = await fetchModelConfig(value.model.key);
    return `${modelConfig.name} (${modelConfig.base.toUpperCase()}) - ${value.weight}`;
  } catch {
    return `${value.model.key} (${value.model.base.toUpperCase()}) - ${value.weight}`;
  }
};

const parameterSetToast = (parameter: string) => {
  toast({
    id: 'PARAMETER_SET',
    title: t('toast.parameterSet'),
    description: t('toast.parameterSetDesc', { parameter }),
    status: 'info',
  });
};

const parameterNotSetToast = (parameter: string, message?: string) => {
  toast({
    id: 'PARAMETER_NOT_SET',
    title: t('toast.parameterNotSet'),
    description: message
      ? t('toast.parameterNotSetDescWithMessage', { parameter, message })
      : t('toast.parameterNotSetDesc', { parameter }),
    status: 'warning',
  });
};

const buildParse =
  <TValue, TItem>(arg: {
    parser: MetadataParseFunc<TValue>;
    getLabel: MetadataGetLabelFunc;
  }): MetadataHandlers<TValue, TItem>['parse'] =>
  async (metadata, withToast = false) => {
    try {
      const parsed = await arg.parser(metadata);
      withToast && parameterSetToast(arg.getLabel());
      return parsed;
    } catch (e) {
      withToast && parameterNotSetToast(arg.getLabel(), (e as Error).message);
      throw e;
    }
  };

const buildParseItem =
  <TValue, TItem>(arg: {
    itemParser: MetadataParseFunc<TItem>;
    getLabel: MetadataGetLabelFunc;
  }): MetadataHandlers<TValue, TItem>['parseItem'] =>
  async (item, withToast = false) => {
    try {
      const parsed = await arg.itemParser(item);
      withToast && parameterSetToast(arg.getLabel());
      return parsed;
    } catch (e) {
      withToast && parameterNotSetToast(arg.getLabel(), (e as Error).message);
      throw e;
    }
  };

const buildRecall =
  <TValue, TItem>(arg: {
    recaller: MetadataRecallFunc<TValue>;
    validator?: MetadataValidateFunc<TValue>;
    getLabel: MetadataGetLabelFunc;
  }): NonNullable<MetadataHandlers<TValue, TItem>['recall']> =>
  async (value, withToast = false) => {
    try {
      arg.validator && (await arg.validator(value));
      await arg.recaller(value);
      withToast && parameterSetToast(arg.getLabel());
    } catch (e) {
      withToast && parameterNotSetToast(arg.getLabel(), (e as Error).message);
      throw e;
    }
  };

const buildRecallItem =
  <TValue, TItem>(arg: {
    itemRecaller: MetadataRecallFunc<TItem>;
    itemValidator?: MetadataValidateFunc<TItem>;
    getLabel: MetadataGetLabelFunc;
  }): NonNullable<MetadataHandlers<TValue, TItem>['recallItem']> =>
  async (item, withToast = false) => {
    try {
      arg.itemValidator && (await arg.itemValidator(item));
      await arg.itemRecaller(item);
      withToast && parameterSetToast(arg.getLabel());
    } catch (e) {
      withToast && parameterNotSetToast(arg.getLabel(), (e as Error).message);
      throw e;
    }
  };

const resolveToString = (value: unknown) =>
  new Promise<string>((resolve) => {
    resolve(String(value));
  });

const buildHandlers: BuildMetadataHandlers = ({
  getLabel,
  parser,
  itemParser,
  recaller,
  itemRecaller,
  validator,
  itemValidator,
  renderValue,
  renderItemValue,
  getIsVisible,
}) => ({
  parse: buildParse({ parser, getLabel }),
  parseItem: itemParser ? buildParseItem({ itemParser, getLabel }) : undefined,
  recall: recaller ? buildRecall({ recaller, validator, getLabel }) : undefined,
  recallItem: itemRecaller ? buildRecallItem({ itemRecaller, itemValidator, getLabel }) : undefined,
  getLabel,
  renderValue: renderValue ?? resolveToString,
  renderItemValue: renderItemValue ?? resolveToString,
  getIsVisible,
});

export const handlers = {
  // Misc
  createdBy: buildHandlers({ getLabel: () => t('metadata.createdBy'), parser: parsers.createdBy }),
  generationMode: buildHandlers({ getLabel: () => t('metadata.generationMode'), parser: parsers.generationMode }),

  // Core parameters
  cfgRescaleMultiplier: buildHandlers({
    getLabel: () => t('metadata.cfgRescaleMultiplier'),
    parser: parsers.cfgRescaleMultiplier,
    recaller: recallers.cfgRescaleMultiplier,
  }),
  cfgScale: buildHandlers({
    getLabel: () => t('metadata.cfgScale'),
    parser: parsers.cfgScale,
    recaller: recallers.cfgScale,
  }),
  guidance: buildHandlers({
    getLabel: () => t('metadata.guidance'),
    parser: parsers.guidance,
    recaller: recallers.guidance,
  }),
  height: buildHandlers({ getLabel: () => t('metadata.height'), parser: parsers.height, recaller: recallers.height }),
  negativePrompt: buildHandlers({
    getLabel: () => t('metadata.negativePrompt'),
    parser: parsers.negativePrompt,
    recaller: recallers.negativePrompt,
  }),
  positivePrompt: buildHandlers({
    getLabel: () => t('metadata.positivePrompt'),
    parser: parsers.positivePrompt,
    recaller: recallers.positivePrompt,
  }),
  scheduler: buildHandlers({
    getLabel: () => t('metadata.scheduler'),
    parser: parsers.scheduler,
    recaller: recallers.scheduler,
  }),
  sdxlNegativeStylePrompt: buildHandlers({
    getLabel: () => t('sdxl.negStylePrompt'),
    parser: parsers.sdxlNegativeStylePrompt,
    recaller: recallers.sdxlNegativeStylePrompt,
  }),
  sdxlPositiveStylePrompt: buildHandlers({
    getLabel: () => t('sdxl.posStylePrompt'),
    parser: parsers.sdxlPositiveStylePrompt,
    recaller: recallers.sdxlPositiveStylePrompt,
  }),
  seed: buildHandlers({ getLabel: () => t('metadata.seed'), parser: parsers.seed, recaller: recallers.seed }),
  steps: buildHandlers({ getLabel: () => t('metadata.steps'), parser: parsers.steps, recaller: recallers.steps }),
  strength: buildHandlers({
    getLabel: () => t('metadata.strength'),
    parser: parsers.strength,
    recaller: recallers.strength,
  }),
  width: buildHandlers({ getLabel: () => t('metadata.width'), parser: parsers.width, recaller: recallers.width }),

  // HRF
  hrfEnabled: buildHandlers({
    getLabel: () => t('hrf.metadata.enabled'),
    parser: parsers.hrfEnabled,
    recaller: recallers.hrfEnabled,
  }),
  hrfMethod: buildHandlers({
    getLabel: () => t('hrf.metadata.method'),
    parser: parsers.hrfMethod,
    recaller: recallers.hrfMethod,
  }),
  hrfStrength: buildHandlers({
    getLabel: () => t('hrf.metadata.strength'),
    parser: parsers.hrfStrength,
    recaller: recallers.hrfStrength,
  }),

  // Refiner
  refinerCFGScale: buildHandlers({
    getLabel: () => t('sdxl.cfgScale'),
    parser: parsers.refinerCFGScale,
    recaller: recallers.refinerCFGScale,
  }),
  refinerModel: buildHandlers({
    getLabel: () => t('sdxl.refinermodel'),
    parser: parsers.refinerModel,
    recaller: recallers.refinerModel,
    validator: validators.refinerModel,
    renderValue: renderModelConfigValue,
  }),
  refinerNegativeAestheticScore: buildHandlers({
    getLabel: () => t('sdxl.posAestheticScore'),
    parser: parsers.refinerNegativeAestheticScore,
    recaller: recallers.refinerNegativeAestheticScore,
  }),
  refinerPositiveAestheticScore: buildHandlers({
    getLabel: () => t('sdxl.negAestheticScore'),
    parser: parsers.refinerPositiveAestheticScore,
    recaller: recallers.refinerPositiveAestheticScore,
  }),
  refinerScheduler: buildHandlers({
    getLabel: () => t('sdxl.scheduler'),
    parser: parsers.refinerScheduler,
    recaller: recallers.refinerScheduler,
  }),
  refinerStart: buildHandlers({
    getLabel: () => t('sdxl.refinerStart'),
    parser: parsers.refinerStart,
    recaller: recallers.refinerStart,
  }),
  refinerSteps: buildHandlers({
    getLabel: () => t('sdxl.refinerSteps'),
    parser: parsers.refinerSteps,
    recaller: recallers.refinerSteps,
  }),

  // Models
  model: buildHandlers({
    getLabel: () => t('metadata.model'),
    parser: parsers.mainModel,
    recaller: recallers.model,
    renderValue: renderModelConfigValue,
  }),
  vae: buildHandlers({
    getLabel: () => t('metadata.vae'),
    parser: parsers.vaeModel,
    recaller: recallers.vae,
    renderValue: renderModelConfigValue,
    validator: validators.vaeModel,
  }),

  // Arrays of models
  loras: buildHandlers({
    getLabel: () => t('models.lora'),
    parser: parsers.loras,
    itemParser: parsers.lora,
    recaller: recallers.loras,
    itemRecaller: recallers.lora,
    validator: validators.loras,
    itemValidator: validators.lora,
    renderItemValue: renderLoRAValue,
  }),

  canvasV2Metadata: buildHandlers({
    getLabel: () => t('metadata.canvasV2Metadata'),
    parser: parsers.canvasV2Metadata,
    recaller: recallers.canvasV2Metadata,
  }),
} as const;

type ParsedValue = Awaited<ReturnType<(typeof handlers)[keyof typeof handlers]['parse']>>;
type RecallResults = Partial<Record<keyof typeof handlers, ParsedValue>>;

export const parseAndRecallPrompts = async (metadata: unknown) => {
  const keysToRecall: (keyof typeof handlers)[] = [
    'positivePrompt',
    'negativePrompt',
    'sdxlPositiveStylePrompt',
    'sdxlNegativeStylePrompt',
  ];
  const recalled = await recallKeys(keysToRecall, metadata);
  if (size(recalled) > 0) {
    parameterSetToast(t('metadata.allPrompts'));
  }
};

export const parseAndRecallImageDimensions = (metadata: unknown) => {
  const recalled = recallKeys(['width', 'height'], metadata);
  if (size(recalled) > 0) {
    parameterSetToast(t('metadata.imageDimensions'));
  }
};

// These handlers should be omitted when recalling to control layers
const TO_CONTROL_LAYERS_SKIP_KEYS: (keyof typeof handlers)[] = ['strength'];
// These handlers should be omitted when recalling to the rest of the app
const NOT_TO_CONTROL_LAYERS_SKIP_KEYS: (keyof typeof handlers)[] = [];

export const parseAndRecallAllMetadata = async (
  metadata: unknown,
  toControlLayers: boolean,
  skip: (keyof typeof handlers)[] = []
) => {
  const skipKeys = deepClone(skip);
  if (toControlLayers) {
    skipKeys.push(...TO_CONTROL_LAYERS_SKIP_KEYS);
  } else {
    skipKeys.push(...NOT_TO_CONTROL_LAYERS_SKIP_KEYS);
  }

  const keysToRecall = objectKeys(handlers).filter((key) => !skipKeys.includes(key));
  const recalled = await recallKeys(keysToRecall, metadata);

  if (size(recalled) > 0) {
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
};

/**
 * Recalls a set of keys from metadata.
 * Includes special handling for some metadata where recalling may have side effects. For example, recalling a "style"
 * prompt that is different from the "positive" or "negative" prompt should disable prompt concatenation.
 * @param keysToRecall An array of keys to recall.
 * @param metadata The metadata to recall from
 * @returns A promise that resolves to an object containing the recalled values.
 */
const recallKeys = async (keysToRecall: (keyof typeof handlers)[], metadata: unknown): Promise<RecallResults> => {
  const { dispatch } = getStore();
  const recalled: RecallResults = {};
  // It's possible for some metadata item's recall to clobber the recall of another. For example, the model recall
  // may change the width and height. If we are also recalling the width and height directly, we need to ensure that the
  // model is recalled first, so it doesn't accidentally override the width and height.
  const sortedKeysToRecall = keysToRecall.sort((a) => {
    if (a === 'model') {
      return -1;
    }
    return 0;
  });
  for (const key of sortedKeysToRecall) {
    const { parse, recall } = handlers[key];
    if (!recall) {
      continue;
    }
    try {
      const value = await parse(metadata);
      /* @ts-expect-error The return type of parse and the input type of recall are guaranteed to be compatible. */
      await recall(value);
      recalled[key] = value;
    } catch {
      // no-op
    }
  }

  if (
    (recalled['sdxlPositiveStylePrompt'] && recalled['sdxlPositiveStylePrompt'] !== recalled['positivePrompt']) ||
    (recalled['sdxlNegativeStylePrompt'] && recalled['sdxlNegativeStylePrompt'] !== recalled['negativePrompt'])
  ) {
    // If we set the negative style prompt or positive style prompt, we should disable prompt concat
    dispatch(shouldConcatPromptsChanged(false));
  } else {
    // Otherwise, we should enable prompt concat
    dispatch(shouldConcatPromptsChanged(true));
  }

  return recalled;
};
