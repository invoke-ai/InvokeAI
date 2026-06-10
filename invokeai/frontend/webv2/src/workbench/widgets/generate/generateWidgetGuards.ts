import type { GenerateSettings, GenerateWidgetValues } from '../../generation/types';

export const isGenerateSettings = (values: unknown): values is GenerateSettings => {
  if (!values || typeof values !== 'object') {
    return false;
  }

  const record = values as Record<string, unknown>;
  const hasFiniteNumber = (key: string) => typeof record[key] === 'number' && Number.isFinite(record[key]);

  return (
    typeof record.modelKey === 'string' &&
    typeof record.positivePrompt === 'string' &&
    typeof record.negativePrompt === 'string' &&
    hasFiniteNumber('batchCount') &&
    hasFiniteNumber('width') &&
    hasFiniteNumber('height') &&
    hasFiniteNumber('steps') &&
    hasFiniteNumber('cfgScale') &&
    hasFiniteNumber('cfgRescaleMultiplier') &&
    typeof record.scheduler === 'string' &&
    hasFiniteNumber('seed') &&
    typeof record.shouldRandomizeSeed === 'boolean'
  );
};

export const isGenerateWidgetValues = (values: unknown): values is GenerateWidgetValues => {
  if (!isGenerateSettings(values)) {
    return false;
  }

  const model = (values as unknown as Record<string, unknown>).model;

  return Boolean(model && typeof model === 'object' && (model as Record<string, unknown>).type === 'main');
};
