import type { WorkbenchAction } from '@workbench/workbenchState';

import { describe, expect, it, vi } from 'vitest';

import { createDocumentPatchEntry, DOCUMENT_PATCH_DEFAULT_BYTES } from './documentPatch';

const forward: WorkbenchAction = { direction: 1, type: 'cycleStagedImage' };
const inverse: WorkbenchAction = { direction: -1, type: 'cycleStagedImage' };

describe('createDocumentPatchEntry', () => {
  it('dispatches inverse on undo and forward on redo', () => {
    const dispatch = vi.fn();
    const entry = createDocumentPatchEntry({ dispatch, forward, inverse, label: 'Cycle' });

    entry.undo();
    expect(dispatch).toHaveBeenNthCalledWith(1, inverse);
    entry.redo();
    expect(dispatch).toHaveBeenNthCalledWith(2, forward);
  });

  it('defaults bytes to a small nominal cost, overridable', () => {
    const dispatch = vi.fn();
    const entry = createDocumentPatchEntry({ dispatch, forward, inverse, label: 'Cycle' });
    expect(entry.bytes).toBe(DOCUMENT_PATCH_DEFAULT_BYTES);

    const heavier = createDocumentPatchEntry({ bytes: 4096, dispatch, forward, inverse, label: 'Cycle' });
    expect(heavier.bytes).toBe(4096);
    expect(heavier.label).toBe('Cycle');
  });
});
