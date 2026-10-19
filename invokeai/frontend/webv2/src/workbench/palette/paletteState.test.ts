import { describe, expect, it, vi } from 'vitest';

import {
  changePaletteSelection,
  commitPaletteQuery,
  createInitialPaletteState,
  enterPaletteScope,
  enterPaletteStage,
  replacePaletteQuery,
  returnPaletteToRoot,
} from './paletteState';

describe('palette state transitions', () => {
  it('replaces and commits a query without preserving a stale selection', () => {
    const selected = changePaletteSelection(createInitialPaletteState(), 'row-1');
    const typed = replacePaletteQuery(selected, 'sunset');

    expect(typed).toEqual({ activeRowId: null, debouncedQuery: '', mode: { kind: 'root' }, query: 'sunset' });
    expect(commitPaletteQuery(typed, 'sunset')).toEqual({ ...typed, debouncedQuery: 'sunset' });
  });

  it('enters scopes and stages with complete, non-contradictory state', () => {
    expect(enterPaletteScope('images', 'cats')).toEqual({
      activeRowId: null,
      debouncedQuery: 'cats',
      mode: { kind: 'scoped', providerKey: 'images' },
      query: 'cats',
    });

    const stage = { options: [{ apply: vi.fn(), id: 'dark', isCurrent: true, label: 'Dark' }], title: 'Theme' };
    expect(enterPaletteStage(stage, 'stage:dark')).toEqual({
      activeRowId: 'stage:dark',
      debouncedQuery: '',
      mode: { kind: 'stage', stage },
      query: '',
    });
  });

  it('returns to a fresh root state', () => {
    expect(returnPaletteToRoot()).toEqual(createInitialPaletteState());
  });

  it('preserves state identity when the requested row is already selected', () => {
    const state = changePaletteSelection(createInitialPaletteState(), 'row-1');

    expect(changePaletteSelection(state, 'row-1')).toBe(state);
  });
});
