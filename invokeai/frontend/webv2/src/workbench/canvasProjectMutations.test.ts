import type { CanvasProjectMutation } from './canvasProjectMutations';

import { describe, expect, it } from 'vitest';

import { isHighConfidenceCanvasEdit } from './canvasProjectMutations';

const mutation = (value: Record<string, unknown>): CanvasProjectMutation => value as never;

describe('isHighConfidenceCanvasEdit', () => {
  it('counts content edits: layer add/source updates, bbox, and document resize', () => {
    expect(isHighConfidenceCanvasEdit(mutation({ type: 'addCanvasLayer' }))).toBe(true);
    expect(isHighConfidenceCanvasEdit(mutation({ type: 'updateCanvasLayerSource' }))).toBe(true);
    expect(isHighConfidenceCanvasEdit(mutation({ type: 'updateCanvasLayerConfig' }))).toBe(true);
    expect(isHighConfidenceCanvasEdit(mutation({ type: 'commitStagedImage' }))).toBe(true);
    expect(isHighConfidenceCanvasEdit(mutation({ type: 'setCanvasBbox' }))).toBe(true);
    expect(isHighConfidenceCanvasEdit(mutation({ type: 'resizeCanvasDocument' }))).toBe(true);
  });

  it('counts layer patches only when they carry content intent', () => {
    expect(isHighConfidenceCanvasEdit(mutation({ patch: { transform: {} }, type: 'updateCanvasLayer' }))).toBe(true);
    expect(isHighConfidenceCanvasEdit(mutation({ patch: { opacity: 0.5 }, type: 'updateCanvasLayer' }))).toBe(true);
    expect(isHighConfidenceCanvasEdit(mutation({ patch: { blendMode: 'multiply' }, type: 'updateCanvasLayer' }))).toBe(
      true
    );
    expect(
      isHighConfidenceCanvasEdit(mutation({ patch: { isLocked: true, name: 'Layer 1' }, type: 'updateCanvasLayer' }))
    ).toBe(false);
  });

  it('ignores selection, visibility, staging review, snapshots, and hydration', () => {
    const excluded: string[] = [
      'setCanvasSelectedLayer',
      'setCanvasLayersEnabled',
      'setStagedImageIndex',
      'cycleStagedImage',
      'discardSelectedStagedImage',
      'discardAllStagedImages',
      'clearCanvasStaging',
      'toggleCanvasStagingVisibility',
      'toggleCanvasStagingThumbnailsVisibility',
      'setCanvasStagingAutoSwitch',
      'rollbackStagedImageCommit',
      'saveCanvasSnapshot',
      'restoreCanvasSnapshot',
      'deleteCanvasSnapshot',
      'replaceCanvasDocument',
    ];

    for (const type of excluded) {
      expect(isHighConfidenceCanvasEdit(mutation({ type }))).toBe(false);
    }
  });
});
