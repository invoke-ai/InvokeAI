import { CONTROL_ADAPTER_DEFAULTS } from '@workbench/controlAdapters';
import { describe, expect, it } from 'vitest';

import { createControlLayer, nextControlLayerName } from './layerFactories';

describe('createControlLayer', () => {
  it('creates an empty, untyped, model-less layer on the legacy ControlNet defaults with transparency on', () => {
    const layer = createControlLayer({ id: 'c1', name: 'Control Layer 1' });
    expect(layer).toEqual({
      adapter: { beginEndStepPct: [0, 0.75], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 0.75 },
      blendMode: 'normal',
      id: 'c1',
      isEnabled: true,
      isLocked: false,
      name: 'Control Layer 1',
      opacity: 1,
      source: { bitmap: null, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'control',
      withTransparencyEffect: true,
    });
    // The defaults' mutable step range is copied, never shared.
    expect(layer.adapter.beginEndStepPct).not.toBe(CONTROL_ADAPTER_DEFAULTS.controlnet.beginEndStepPct);
  });

  it.each([
    ['anima', 'anima_lllite'],
    ['z-image', 'z_image_control'],
    ['sdxl', 'controlnet'],
  ] as const)('starts a %s layer on its kind’s defaults without a model', (base, kind) => {
    expect(createControlLayer({ base, id: 'c1', name: 'Control Layer 1' }).adapter).toEqual({
      ...CONTROL_ADAPTER_DEFAULTS[kind],
      model: null,
    });
  });

  it('uses the given pixel source', () => {
    const source = { image: { height: 48, imageName: 'photo.png', width: 64 }, type: 'image' } as const;
    expect(createControlLayer({ id: 'c1', name: 'C', source }).source).toBe(source);
  });
});

describe('nextControlLayerName', () => {
  it('returns the first free "Control Layer N" gap', () => {
    expect(nextControlLayerName([])).toBe('Control Layer 1');
    expect(nextControlLayerName(['Control Layer 1', 'Control Layer 3', 'Layer 2'])).toBe('Control Layer 2');
    expect(nextControlLayerName(['Control Layer 1', 'Control Layer 2'])).toBe('Control Layer 3');
  });
});
