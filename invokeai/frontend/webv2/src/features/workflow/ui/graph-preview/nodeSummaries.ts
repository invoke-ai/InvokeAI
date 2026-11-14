import type { WorkflowPreviewValue } from '@features/workflow/core/workflowPreview';
import type { TFunction } from 'i18next';

/** Return null when no type-specific summary exists so callers use the generic node/input summary. */
export const getNodeSubtitle = (
  node: { id: string; type: string; inputs: Record<string, unknown> },
  t: TFunction
): string | null => {
  if (node.type === 'denoise_latents') {
    const { cfg_scale: cfg, steps } = node.inputs;

    if (typeof cfg === 'number' && typeof steps === 'number') {
      return t('graphPreview.nodeSummary.denoise', { cfg, count: steps });
    }

    return null;
  }

  if (node.type === 'noise') {
    const { height, width } = node.inputs;

    if (typeof height === 'number' && typeof width === 'number') {
      return t('graphPreview.nodeSummary.noise', { height, width });
    }

    return null;
  }

  return null;
};

const FRAME_FIELDS = ['num_frames', 'frames', 'frame_count'] as const;

const getNumber = (values: Readonly<Record<string, unknown>>, name: string): number | null => {
  const value = values[name];

  return typeof value === 'number' && Number.isFinite(value) ? value : null;
};

const getModelName = (values: Readonly<Record<string, unknown>>): string | null => {
  for (const value of Object.values(values)) {
    if (typeof value === 'object' && value !== null && typeof (value as { key?: unknown }).key === 'string') {
      const { name } = value as { name?: unknown };

      if (typeof name === 'string' && name) {
        return name;
      }
    }
  }

  return null;
};

/**
 * A saved workflow node's second line in the library preview, from the values it holds itself: steps (with frames or
 * CFG), a size, or the model it names. Null when none applies; the node then shows its title alone.
 */
export const getWorkflowNodeSummary = (values: Readonly<Record<string, unknown>>, t: TFunction): string | null => {
  const steps = getNumber(values, 'steps');

  if (steps !== null) {
    const frames = FRAME_FIELDS.map((name) => getNumber(values, name)).find((value) => value !== null) ?? null;
    const cfg = getNumber(values, 'cfg_scale');

    return frames !== null
      ? t('workflowLibrary.preview.summaryFramesSteps', {
          frames: t('workflowLibrary.preview.summaryFrames', { count: frames }),
          steps: t('workflowLibrary.preview.summarySteps', { count: steps }),
        })
      : cfg !== null
        ? t('graphPreview.nodeSummary.denoise', { cfg, count: steps })
        : t('workflowLibrary.preview.summarySteps', { count: steps });
  }

  const width = getNumber(values, 'width');
  const height = getNumber(values, 'height');

  if (width !== null && height !== null) {
    return t('graphPreview.nodeSummary.noise', { height, width });
  }

  return getModelName(values);
};

/** A preview value as the library shows it: a model or image by name, a list by its length, a switch as on or off. */
export const formatWorkflowPreviewValue = (value: WorkflowPreviewValue, t: TFunction): string => {
  switch (value.kind) {
    case 'boolean':
      return value.value ? t('workflowLibrary.preview.valueOn') : t('workflowLibrary.preview.valueOff');
    case 'list':
      return t('workflowLibrary.preview.valueItems', { count: value.count });
    case 'number':
      return t('workflowLibrary.preview.valueNumber', { value: value.value });
    case 'image':
    case 'model':
      return value.name;
    case 'text':
      return value.text;
  }
};
