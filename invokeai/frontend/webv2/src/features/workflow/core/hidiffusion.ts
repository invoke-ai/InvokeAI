import type { FieldInputTemplate, WorkflowFieldInstance, WorkflowInvocationNode } from './types';

export const isHiDiffusionNode = (type: string): boolean =>
  type === 'denoise_latents' || type === 'denoise_latents_meta';

/** Translate legacy nullable ratios before fresh template defaults can hide their manual intent. */
export const normalizeHiDiffusionInputs = (
  inputs: Record<string, WorkflowFieldInstance>,
  original: Record<string, Pick<WorkflowFieldInstance, 'value'>> = inputs,
  connectedFields: ReadonlySet<string> = new Set()
): void => {
  const automatic = original.hidiffusion_auto_ratios?.value;
  const auto =
    typeof automatic === 'boolean'
      ? automatic
      : original.hidiffusion_t1_ratio?.value === null &&
        original.hidiffusion_t2_ratio?.value === null &&
        !connectedFields.has('hidiffusion_t1_ratio') &&
        !connectedFields.has('hidiffusion_t2_ratio');
  inputs.hidiffusion_auto_ratios = { ...inputs.hidiffusion_auto_ratios, name: 'hidiffusion_auto_ratios', value: auto };
  for (const name of ['hidiffusion_t1_ratio', 'hidiffusion_t2_ratio']) {
    const value = original[name]?.value;
    inputs[name] = {
      ...inputs[name],
      name,
      value: value === null ? 0 : (value ?? inputs[name]?.value ?? (name === 'hidiffusion_t1_ratio' ? 0.4 : 0)),
    };
  }
};

export const isHiDiffusionRatioField = (name: string): boolean =>
  name === 'hidiffusion_t1_ratio' || name === 'hidiffusion_t2_ratio';

export const areHiDiffusionRatiosAutomatic = (node: WorkflowInvocationNode): boolean => {
  const auto = node.data.inputs.hidiffusion_auto_ratios?.value;
  return typeof auto === 'boolean'
    ? auto
    : node.data.inputs.hidiffusion_t1_ratio?.value === null && node.data.inputs.hidiffusion_t2_ratio?.value === null;
};

export const getHiDiffusionRatioOrderIssue = (
  node: WorkflowInvocationNode,
  connectedFields: ReadonlySet<string>
): string | null => {
  if (
    !isHiDiffusionNode(node.data.type) ||
    areHiDiffusionRatiosAutomatic(node) ||
    node.data.inputs.hidiffusion?.value !== true ||
    node.data.inputs.hidiffusion_raunet?.value === false ||
    ['hidiffusion', 'hidiffusion_raunet', 'hidiffusion_t1_ratio', 'hidiffusion_t2_ratio'].some((name) =>
      connectedFields.has(name)
    )
  ) {
    return null;
  }
  const t1 = node.data.inputs.hidiffusion_t1_ratio?.value ?? 0;
  const t2 = node.data.inputs.hidiffusion_t2_ratio?.value ?? 0;
  return typeof t1 === 'number' && typeof t2 === 'number' && t2 > t1
    ? 'HiDiffusion T2 ratio must be less than or equal to the T1 ratio.'
    : null;
};

export const getHiDiffusionRatioTemplate = (
  node: WorkflowInvocationNode,
  template: FieldInputTemplate,
  connectedFields: ReadonlySet<string>
): FieldInputTemplate => {
  if (
    !isHiDiffusionNode(node.data.type) ||
    !isHiDiffusionRatioField(template.name) ||
    areHiDiffusionRatiosAutomatic(node) ||
    node.data.inputs.hidiffusion?.value !== true ||
    node.data.inputs.hidiffusion_raunet?.value === false
  ) {
    return template;
  }
  const companion = template.name === 'hidiffusion_t1_ratio' ? 'hidiffusion_t2_ratio' : 'hidiffusion_t1_ratio';
  const value = node.data.inputs[companion]?.value;
  if (connectedFields.has(companion) || typeof value !== 'number') {
    return template;
  }
  return template.name === 'hidiffusion_t1_ratio'
    ? { ...template, minimum: Math.max(template.minimum ?? 0, value) }
    : { ...template, maximum: Math.min(template.maximum ?? 1, value) };
};
