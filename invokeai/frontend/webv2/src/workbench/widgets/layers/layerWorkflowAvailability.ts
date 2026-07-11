import type { InvocationTemplatesSnapshot } from '@workbench/workflows/templates';
import type { ProjectGraphState } from '@workbench/workflows/types';

import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import {
  getDefaultLayerWorkflowSelection,
  getLayerWorkflowInputs,
  getLayerWorkflowOutputs,
  getRunnableLayerWorkflowInputs,
  reconcileLayerWorkflowSelection,
  type GetRunnableLayerWorkflowInputs,
  type LayerWorkflowSelection,
  type WorkflowImageBinding,
} from '@workbench/workflows/layerWorkflow';
import { useInvocationTemplatesSnapshot } from '@workbench/workflows/templates';

import type { LayerWorkflowFailureStage } from './runLayerWorkflow';

const EMPTY_BINDINGS: readonly WorkflowImageBinding[] = [];

export const layerWorkflowBindingKey = (binding: WorkflowImageBinding): string =>
  JSON.stringify([binding.nodeId, binding.fieldName]);

export interface LayerWorkflowAvailability {
  canRunWorkflow: boolean;
  document: ProjectGraphState;
  getRunnableInputs(output: WorkflowImageBinding): readonly WorkflowImageBinding[];
  hasWorkflowBindings: boolean;
  inputs: readonly WorkflowImageBinding[];
  outputs: readonly WorkflowImageBinding[];
  templatesSnapshot: InvocationTemplatesSnapshot;
}

const availabilityCache = new WeakMap<
  ProjectGraphState,
  WeakMap<InvocationTemplatesSnapshot, LayerWorkflowAvailability>
>();

const getLayerWorkflowAvailability = (
  document: ProjectGraphState,
  templatesSnapshot: InvocationTemplatesSnapshot
): LayerWorkflowAvailability => {
  let byTemplates = availabilityCache.get(document);
  if (!byTemplates) {
    byTemplates = new WeakMap();
    availabilityCache.set(document, byTemplates);
  }
  const cached = byTemplates.get(templatesSnapshot);
  if (cached) {
    return cached;
  }

  const inputs =
    templatesSnapshot.status === 'loaded'
      ? getLayerWorkflowInputs(document, templatesSnapshot.templates)
      : EMPTY_BINDINGS;
  const outputs =
    templatesSnapshot.status === 'loaded'
      ? getLayerWorkflowOutputs(document, templatesSnapshot.templates)
      : EMPTY_BINDINGS;
  const runnableByOutput = new Map<string, readonly WorkflowImageBinding[]>();
  for (const output of outputs) {
    runnableByOutput.set(
      layerWorkflowBindingKey(output),
      getRunnableLayerWorkflowInputs(document, templatesSnapshot, output)
    );
  }
  const availability: LayerWorkflowAvailability = {
    canRunWorkflow: [...runnableByOutput.values()].some((runnableInputs) => runnableInputs.length > 0),
    document,
    getRunnableInputs: (output) => runnableByOutput.get(layerWorkflowBindingKey(output)) ?? EMPTY_BINDINGS,
    hasWorkflowBindings: inputs.length > 0 && outputs.length > 0,
    inputs,
    outputs,
    templatesSnapshot,
  };
  byTemplates.set(templatesSnapshot, availability);
  return availability;
};

export const useLayerWorkflowAvailability = (): LayerWorkflowAvailability => {
  const document = useActiveProjectSelector((project) => project.projectGraph, Object.is);
  const templatesSnapshot = useInvocationTemplatesSnapshot();
  return getLayerWorkflowAvailability(document, templatesSnapshot);
};

export const createDefaultLayerWorkflowSelection = (availability: LayerWorkflowAvailability): LayerWorkflowSelection =>
  getDefaultLayerWorkflowSelection(availability.outputs, availability.getRunnableInputs);

export const reconcileLayerWorkflowOperationSelection = (
  current: LayerWorkflowSelection,
  outputs: readonly WorkflowImageBinding[],
  getRunnableInputs: GetRunnableLayerWorkflowInputs
): LayerWorkflowSelection => {
  const selectedOutput = current.output;
  const currentOutput = selectedOutput
    ? outputs.find((output) => layerWorkflowBindingKey(output) === layerWorkflowBindingKey(selectedOutput))
    : undefined;
  if (currentOutput) {
    return reconcileLayerWorkflowSelection(current, currentOutput, getRunnableInputs(currentOutput));
  }
  return {
    ...getDefaultLayerWorkflowSelection(outputs, getRunnableInputs),
    destination: current.destination,
  };
};

export const getLayerWorkflowFailureKey = (stage: LayerWorkflowFailureStage): string => {
  switch (stage) {
    case 'graph':
      return 'widgets.layers.runWorkflow.graphFailure';
    case 'hydrate':
      return 'widgets.layers.runWorkflow.hydrationFailure';
    case 'durability':
      return 'widgets.layers.runWorkflow.durabilityFailure';
    default:
      return 'widgets.layers.runWorkflow.failed';
  }
};
