import { describe, expect, it } from 'vitest';

import { createEmptyCanvasState } from './canvasMigration';
import { gateProjectCanvases } from './projectCanvasGate';

const canvas = createEmptyCanvasState();

describe('gateProjectCanvases', () => {
  it('admits a project whose live canvas loads and ignores removed queue history', () => {
    expect(gateProjectCanvases({ canvas, id: 'p', name: 'P', queue: { items: [{ snapshot: null }] } })).toBeNull();
    expect(gateProjectCanvases({ id: 'legacy', name: 'Legacy' })).toBeNull();
    expect(gateProjectCanvases('not a project')).toBeNull();
  });

  it('refuses a project whose live canvas is unsupported or invalid, keeping the raw document', () => {
    const future = { canvas: { ...canvas, version: 5 }, id: 'p', name: 'P' };
    const broken = { canvas: { ...canvas, version: '2' }, id: 'p', name: 'P' };

    expect(gateProjectCanvases(future)).toMatchObject({
      projectId: 'p',
      projectName: 'P',
      raw: future,
      refusal: { scope: 'state', status: 'unsupported-version', version: 5 },
      source: 'canvas',
    });
    expect(gateProjectCanvases(broken)).toMatchObject({
      raw: broken,
      refusal: { status: 'invalid' },
      source: 'canvas',
    });
  });
});
