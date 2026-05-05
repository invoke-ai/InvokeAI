import { describe, expect, it } from 'vitest';

import { getIsCanvasMergeDownHotkeyEnabled, getIsCanvasMergeVisibleHotkeyEnabled } from './canvasMergeHotkeyUtils';

describe('canvas merge hotkey gating', () => {
  const selectedEntityIdentifier = { id: 'selected', type: 'raster_layer' } as const;
  const entityIdentifierBelowThisOne = { id: 'below', type: 'raster_layer' } as const;

  describe('getIsCanvasMergeDownHotkeyEnabled', () => {
    it('returns false when nothing is selected', () => {
      expect(getIsCanvasMergeDownHotkeyEnabled(null, entityIdentifierBelowThisOne, false)).toBe(false);
    });

    it('returns false when there is no entity below the selection', () => {
      expect(getIsCanvasMergeDownHotkeyEnabled(selectedEntityIdentifier, null, false)).toBe(false);
    });

    it('returns false when the canvas is busy', () => {
      expect(getIsCanvasMergeDownHotkeyEnabled(selectedEntityIdentifier, entityIdentifierBelowThisOne, true)).toBe(
        false
      );
    });

    it('returns true when the selection can be merged down', () => {
      expect(getIsCanvasMergeDownHotkeyEnabled(selectedEntityIdentifier, entityIdentifierBelowThisOne, false)).toBe(
        true
      );
    });
  });

  describe('getIsCanvasMergeVisibleHotkeyEnabled', () => {
    it('returns false when nothing is selected', () => {
      expect(getIsCanvasMergeVisibleHotkeyEnabled(null, 2, false)).toBe(false);
    });

    it('returns false when there are not enough visible entities', () => {
      expect(getIsCanvasMergeVisibleHotkeyEnabled(selectedEntityIdentifier, 1, false)).toBe(false);
    });

    it('returns false when the canvas is busy', () => {
      expect(getIsCanvasMergeVisibleHotkeyEnabled(selectedEntityIdentifier, 2, true)).toBe(false);
    });

    it('returns true when visible entities can be merged', () => {
      expect(getIsCanvasMergeVisibleHotkeyEnabled(selectedEntityIdentifier, 2, false)).toBe(true);
    });
  });
});
