import type { PaletteStage } from './entries';

export type PaletteMode =
  | { kind: 'root' }
  | { kind: 'scoped'; providerKey: string }
  | { kind: 'stage'; stage: PaletteStage };

export interface PaletteState {
  mode: PaletteMode;
  query: string;
  debouncedQuery: string;
  activeRowId: string | null;
}

export const createInitialPaletteState = (): PaletteState => ({
  activeRowId: null,
  debouncedQuery: '',
  mode: { kind: 'root' },
  query: '',
});

export const replacePaletteQuery = (state: PaletteState, query: string): PaletteState => ({
  ...state,
  activeRowId: null,
  query,
});

export const commitPaletteQuery = (state: PaletteState, debouncedQuery: string): PaletteState => ({
  ...state,
  debouncedQuery,
});

export const enterPaletteScope = (providerKey: string, query: string): PaletteState => ({
  activeRowId: null,
  debouncedQuery: query,
  mode: { kind: 'scoped', providerKey },
  query,
});

export const enterPaletteStage = (stage: PaletteStage, activeRowId: string | null): PaletteState => ({
  activeRowId,
  debouncedQuery: '',
  mode: { kind: 'stage', stage },
  query: '',
});

export const returnPaletteToRoot = (): PaletteState => createInitialPaletteState();

export const changePaletteSelection = (state: PaletteState, activeRowId: string | null): PaletteState => ({
  ...state,
  activeRowId,
});
