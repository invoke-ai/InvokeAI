import {
  documentFrom,
  groupContract,
  layerContract,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { describe, expect, it } from 'vitest';

import { selectBlendTarget } from './LayerBlendRow';

describe('selectBlendTarget', () => {
  const projectFor = (nodes: Parameters<typeof documentFrom>[0], selectedLayerId: string | null) => ({
    canvas: { document: documentFrom(nodes, selectedLayerId) },
  });

  it('targets a selected leaf and a selected raster-stack group', () => {
    const leaf = layerContract('r1');
    const group = groupContract('g', [layerContract('r2')], { blendMode: 'multiply', opacity: 0.5 });
    expect(selectBlendTarget(projectFor([leaf, group], 'r1'))).toBe(leaf);
    expect(selectBlendTarget(projectFor([leaf, group], 'g'))).toBe(group);
  });

  it('returns null for an overlay-stack group, a missing id, and no selection', () => {
    const overlayGroup = groupContract('og', [layerContract('c1', 'control')]);
    expect(selectBlendTarget(projectFor([overlayGroup], 'og'))).toBeNull();
    expect(selectBlendTarget(projectFor([layerContract('r1')], 'gone'))).toBeNull();
    expect(selectBlendTarget(projectFor([layerContract('r1')], null))).toBeNull();
  });
});
