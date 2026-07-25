import type { ModelConfig } from '@features/models';
import type { CanvasControlLayerContract } from '@workbench/canvas-engine/api';

import { describe, expect, it } from 'vitest';

import {
  getBlockingControlLayerIssues,
  getControlLayerAttentionReason,
  hasControlLayerContent,
} from './controlLayerChecks';

const model = (key: string, base: string, type: string): ModelConfig => ({ base, key, name: key, type }) as ModelConfig;

const controlLayer = (
  id: string,
  overrides: {
    isEnabled?: boolean;
    kind?: CanvasControlLayerContract['adapter']['kind'];
    model?: string | null;
    source?: CanvasControlLayerContract['source'];
    weight?: number;
  } = {}
): CanvasControlLayerContract => ({
  adapter: {
    beginEndStepPct: [0, 1],
    controlMode: null,
    kind: overrides.kind ?? 'controlnet',
    model: overrides.model ?? null,
    weight: overrides.weight ?? 1,
  },
  blendMode: 'normal',
  id,
  isEnabled: overrides.isEnabled ?? true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: overrides.source ?? { image: { height: 64, imageName: `${id}.png`, width: 64 }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect: true,
});

const sdxl = { base: 'sdxl' };
const models = [model('sdxl-control', 'sdxl', 'controlnet'), model('sd1-control', 'sd-1', 'controlnet')];

describe('hasControlLayerContent', () => {
  it('matches the pipeline content gate', () => {
    expect(hasControlLayerContent(controlLayer('image'))).toBe(true);
    expect(
      hasControlLayerContent(
        controlLayer('painted', { source: { bitmap: { height: 8, width: 8 } as never, type: 'paint' } })
      )
    ).toBe(true);
    expect(hasControlLayerContent(controlLayer('empty', { source: { bitmap: null, type: 'paint' } }))).toBe(false);
  });
});

describe('getBlockingControlLayerIssues', () => {
  it('reports a content-bearing model-less control layer with a human sentence', () => {
    const issues = getBlockingControlLayerIssues({
      layers: [controlLayer('Control Layer 1')],
      mainModel: sdxl,
      models,
    });

    expect(issues).toEqual([
      {
        code: 'missing_model',
        layerId: 'Control Layer 1',
        layerName: 'Control Layer 1',
        message: 'Control layer "Control Layer 1" has no control model selected.',
      },
    ]);
  });

  it('skips disabled, content-less, and non-control layers', () => {
    const issues = getBlockingControlLayerIssues({
      layers: [
        controlLayer('disabled', { isEnabled: false }),
        controlLayer('empty', { source: { bitmap: null, type: 'paint' } }),
      ],
      mainModel: sdxl,
      models,
    });

    expect(issues).toEqual([]);
  });

  it('reports an incompatible base model', () => {
    const issues = getBlockingControlLayerIssues({
      layers: [controlLayer('mismatch', { model: 'sd1-control' })],
      mainModel: sdxl,
      models,
    });

    expect(issues).toMatchObject([{ code: 'incompatible_base' }]);
  });

  it('advances the control LoRA limit counter only past valid layers', () => {
    const fluxModels = [model('flux-lora', 'flux', 'control_lora')];
    const issues = getBlockingControlLayerIssues({
      layers: [
        controlLayer('first', { kind: 'control_lora', model: 'flux-lora' }),
        controlLayer('second', { kind: 'control_lora', model: 'flux-lora' }),
      ],
      mainModel: { base: 'flux' },
      models: fluxModels,
    });

    expect(issues).toEqual([
      {
        code: 'control_lora_limit',
        layerId: 'second',
        layerName: 'second',
        message: 'Only one Control LoRA can be used at a time.',
      },
    ]);
  });

  it('reports issues in document order', () => {
    const issues = getBlockingControlLayerIssues({
      layers: [controlLayer('top'), controlLayer('bottom')],
      mainModel: sdxl,
      models,
    });

    expect(issues.map((issue) => issue.layerName)).toEqual(['top', 'bottom']);
  });
});

describe('getControlLayerAttentionReason', () => {
  it('warns about a missing model even without content', () => {
    const layer = controlLayer('fresh', { source: { bitmap: null, type: 'paint' } });

    expect(getControlLayerAttentionReason(layer, 'sdxl', models)).toBe('missing_model');
  });

  it('returns null for a healthy layer and ignores per-kind limits', () => {
    expect(getControlLayerAttentionReason(controlLayer('ok', { model: 'sdxl-control' }), 'sdxl', models)).toBeNull();
  });
});
