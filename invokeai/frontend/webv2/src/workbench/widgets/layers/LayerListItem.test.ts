import { describe, expect, it } from 'vitest';

import { getLayerListItemInteractionState } from './LayerListItem';

describe('getLayerListItemInteractionState', () => {
  it('keeps selection available while disabling document edits and sorting during an operation', () => {
    expect(getLayerListItemInteractionState(true)).toEqual({
      canRename: false,
      canSelect: true,
      canToggleLock: false,
      canToggleVisibility: false,
      sortableDisabled: true,
    });
  });

  it('enables ordinary row editing and sorting when document editing is unlocked', () => {
    expect(getLayerListItemInteractionState(false)).toEqual({
      canRename: true,
      canSelect: true,
      canToggleLock: true,
      canToggleVisibility: true,
      sortableDisabled: false,
    });
  });
});
