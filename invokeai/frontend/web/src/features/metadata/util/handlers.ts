import { objectKeys } from 'common/util/objectKeys';
import { toast } from 'common/util/toast';
import type { LoRA } from 'features/lora/store/loraSlice';
import type {
  AnyControlAdapterConfigMetadata,
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
import { t } from 'i18next';

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
const renderControlAdapterValue: MetadataRenderValueFunc<AnyControlAdapterConfigMetadata> = async (value) => {
  try {
    const modelConfig = await fetchModelConfig(value.model.key ?? 'none');
    return `${modelConfig.name} (${modelConfig.base.toUpperCase()}) - ${value.weight}`;
  } catch {
    return `${value.model.key} (${value.model.base.toUpperCase()}) - ${value.weight}`;
  }
};

const parameterSetToast = (parameter: string, description?: string) => {
  toast({
    title: t('toast.parameterSet', { parameter }),
    description,
    status: 'info',
    duration: 2500,
    isClosable: true,
  });
};

const parameterNotSetToast = (parameter: string, description?: string) => {
  toast({
    title: t('toast.parameterNotSet', { parameter }),
    description,
    status: 'warning',
    duration: 2500,
    isClosable: true,
  });
};

// const allParameterSetToast = (description?: string) => {
//   toast({
//     title: t('toast.parametersSet'),
//     status: 'info',
//     description,
//     duration: 2500,
//     isClosable: true,
//   });
// };

// const allParameterNotSetToast = (description?: string) => {
//   toast({
//     title: t('toast.parametersNotSet'),
//     status: 'warning',
//     description,
//     duration: 2500,
//     isClosable: true,
//   });
// };

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

const resolveToString = (value: unknown) => new Promise<string>((resolve) => resolve(String(value)));

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
}) => ({
  parse: buildParse({ parser, getLabel }),
  parseItem: itemParser ? buildParseItem({ itemParser, getLabel }) : undefined,
  recall: recaller ? buildRecall({ recaller, validator, getLabel }) : undefined,
  recallItem: itemRecaller ? buildRecallItem({ itemRecaller, itemValidator, getLabel }) : undefined,
  getLabel,
  renderValue: renderValue ?? resolveToString,
  renderItemValue: renderItemValue ?? resolveToString,
});

export const handlers = {
  // Misc
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
    getLabel: () => t('sdxl.refinerModel'),
    parser: parsers.refinerModel,
    recaller: recallers.refinerModel,
    validator: validators.refinerModel,
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
    getLabel: () => t('sdxl.refiner_start'),
    parser: parsers.refinerStart,
    recaller: recallers.refinerStart,
  }),
  refinerSteps: buildHandlers({
    getLabel: () => t('sdxl.refiner_steps'),
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
  controlNets: buildHandlers({
    getLabel: () => t('common.controlNet'),
    parser: parsers.controlNets,
    itemParser: parsers.controlNet,
    recaller: recallers.controlNets,
    itemRecaller: recallers.controlNet,
    validator: validators.controlNets,
    itemValidator: validators.controlNet,
    renderItemValue: renderControlAdapterValue,
  }),
  ipAdapters: buildHandlers({
    getLabel: () => t('common.ipAdapter'),
    parser: parsers.ipAdapters,
    itemParser: parsers.ipAdapter,
    recaller: recallers.ipAdapters,
    itemRecaller: recallers.ipAdapter,
    validator: validators.ipAdapters,
    itemValidator: validators.ipAdapter,
    renderItemValue: renderControlAdapterValue,
  }),
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
  t2iAdapters: buildHandlers({
    getLabel: () => t('common.t2iAdapter'),
    parser: parsers.t2iAdapters,
    itemParser: parsers.t2iAdapter,
    recaller: recallers.t2iAdapters,
    itemRecaller: recallers.t2iAdapter,
    validator: validators.t2iAdapters,
    itemValidator: validators.t2iAdapter,
    renderItemValue: renderControlAdapterValue,
  }),
} as const;

export const parseAndRecallPrompts = async (metadata: unknown) => {
  const results = await Promise.allSettled([
    handlers.positivePrompt.parse(metadata).then((positivePrompt) => {
      if (!handlers.positivePrompt.recall) {
        return;
      }
      handlers.positivePrompt?.recall(positivePrompt);
    }),
    handlers.negativePrompt.parse(metadata).then((negativePrompt) => {
      if (!handlers.negativePrompt.recall) {
        return;
      }
      handlers.negativePrompt?.recall(negativePrompt);
    }),
    handlers.sdxlPositiveStylePrompt.parse(metadata).then((sdxlPositiveStylePrompt) => {
      if (!handlers.sdxlPositiveStylePrompt.recall) {
        return;
      }
      handlers.sdxlPositiveStylePrompt?.recall(sdxlPositiveStylePrompt);
    }),
    handlers.sdxlNegativeStylePrompt.parse(metadata).then((sdxlNegativeStylePrompt) => {
      if (!handlers.sdxlNegativeStylePrompt.recall) {
        return;
      }
      handlers.sdxlNegativeStylePrompt?.recall(sdxlNegativeStylePrompt);
    }),
  ]);
  if (results.some((result) => result.status === 'fulfilled')) {
    parameterSetToast(t('metadata.allPrompts'));
  }
};

export const parseAndRecallImageDimensions = async (metadata: unknown) => {
  const results = await Promise.allSettled([
    handlers.width.parse(metadata).then((width) => {
      if (!handlers.width.recall) {
        return;
      }
      handlers.width?.recall(width);
    }),
    handlers.height.parse(metadata).then((height) => {
      if (!handlers.height.recall) {
        return;
      }
      handlers.height?.recall(height);
    }),
  ]);
  if (results.some((result) => result.status === 'fulfilled')) {
    parameterSetToast(t('metadata.imageDimensions'));
  }
};

export const parseAndRecallAllMetadata = async (metadata: unknown, skip: (keyof typeof handlers)[] = []) => {
  const results = await Promise.allSettled(
    objectKeys(handlers)
      .filter((key) => !skip.includes(key))
      .map((key) => {
        const { parse, recall } = handlers[key];
        return parse(metadata).then((value) => {
          if (!recall) {
            return;
          }
          /* @ts-expect-error The return type of parse and the input type of recall are guaranteed to be compatible. */
          recall(value);
        });
      })
  );
  if (results.some((result) => result.status === 'fulfilled')) {
    parameterSetToast(t('toast.parametersSet'));
  }
};
