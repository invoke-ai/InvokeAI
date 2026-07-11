import type { WorkflowImageBinding } from '@workbench/workflows/layerWorkflow';

import { describe, expect, it } from 'vitest';

import { getLayerWorkflowFailureKey, reconcileLayerWorkflowDialogSelection } from './RunLayerWorkflowDialog';

const binding = (nodeId: string, fieldName = 'image', label = `${nodeId} → ${fieldName}`): WorkflowImageBinding => ({
  fieldName,
  label,
  nodeId,
});

describe('reconcileLayerWorkflowDialogSelection', () => {
  it('preserves a still-runnable input and output by field identity when availability refreshes', () => {
    const refreshedInput = binding('input', 'image', 'Refreshed input');
    const refreshedOutput = binding('output', 'image', 'Refreshed output');

    expect(
      reconcileLayerWorkflowDialogSelection(
        {
          destination: 'staging',
          input: binding('input', 'image', 'Old input'),
          output: binding('output', 'image', 'Old output'),
        },
        [refreshedOutput],
        () => [refreshedInput]
      )
    ).toEqual({ destination: 'staging', input: refreshedInput, output: refreshedOutput });
  });

  it('keeps the selected output and clears its input when that output has no runnable inputs', () => {
    const selectedOutput = binding('selected-output');
    const fallbackOutput = binding('fallback-output');
    const fallbackInput = binding('fallback-input');

    expect(
      reconcileLayerWorkflowDialogSelection(
        { destination: 'copy-raster', input: binding('stale-input'), output: selectedOutput },
        [selectedOutput, fallbackOutput],
        (output) => (output.nodeId === fallbackOutput.nodeId ? [fallbackInput] : [])
      )
    ).toEqual({ destination: 'copy-raster', input: null, output: selectedOutput });
  });

  it('falls back to the first runnable pair when the selected output disappeared', () => {
    const fallbackOutput = binding('fallback-output');
    const fallbackInput = binding('fallback-input');

    expect(
      reconcileLayerWorkflowDialogSelection(
        { destination: 'replace', input: binding('stale-input'), output: binding('removed-output') },
        [fallbackOutput],
        () => [fallbackInput]
      )
    ).toEqual({ destination: 'replace', input: fallbackInput, output: fallbackOutput });
  });
});

describe('getLayerWorkflowFailureKey', () => {
  it.each([
    ['graph', 'widgets.layers.runWorkflow.graphFailure'],
    ['hydrate', 'widgets.layers.runWorkflow.hydrationFailure'],
    ['durability', 'widgets.layers.runWorkflow.durabilityFailure'],
    ['export', 'widgets.layers.runWorkflow.failed'],
    ['upload', 'widgets.layers.runWorkflow.failed'],
    ['gallery', 'widgets.layers.runWorkflow.failed'],
    ['staging', 'widgets.layers.runWorkflow.failed'],
    ['commit', 'widgets.layers.runWorkflow.failed'],
  ] as const)('maps %s failures to %s', (stage, key) => {
    expect(getLayerWorkflowFailureKey(stage)).toBe(key);
  });
});
