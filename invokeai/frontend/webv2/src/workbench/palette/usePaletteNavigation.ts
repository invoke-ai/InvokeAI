import type { Dispatch, KeyboardEvent, SetStateAction } from 'react';

import { useCallback, useMemo } from 'react';

import type { PaletteEntry, PaletteRow, PaletteStage } from './entries';
import type { PaletteState } from './paletteState';

import { resolveActivePaletteRow, STAGE_ENTRY_ID_PREFIX } from './entries';
import { changePaletteSelection, enterPaletteStage } from './paletteState';
import { recordRecentEntry } from './recents';

const PAGE_JUMP_ROWS = 8;

export type ScrollToIndex = (index: number) => void;

export const usePaletteNavigation = ({
  activeRowId,
  clearDebounce,
  enterScope,
  focusInput,
  onClose,
  popOverlay,
  previewRow,
  query,
  rows,
  scopeProviderKey,
  setPaletteState,
  stage,
}: {
  activeRowId: string | null;
  clearDebounce: () => void;
  enterScope: (providerKey: string) => void;
  focusInput: () => void;
  onClose: () => void;
  popOverlay: () => void;
  previewRow: (row: PaletteRow | undefined) => void;
  query: string;
  rows: PaletteRow[];
  scopeProviderKey: string | null;
  setPaletteState: Dispatch<SetStateAction<PaletteState>>;
  stage: PaletteStage | null;
}) => {
  const navigableRows = useMemo(
    () => rows.flatMap((row, index) => (row.kind === 'label' ? [] : [{ index, row }])),
    [rows]
  );
  const effectiveActive = useMemo(() => resolveActivePaletteRow(rows, activeRowId), [activeRowId, rows]);
  const activeRow = effectiveActive?.row;

  const runRow = useCallback(
    (row: PaletteRow) => {
      if (row.kind === 'label') {
        return;
      }

      if (row.kind === 'scope') {
        enterScope(row.providerKey);
        return;
      }

      if (row.kind === 'provider-error') {
        row.retry();
        focusInput();
        return;
      }

      if (row.entry.isPersistentRecent) {
        recordRecentEntry(row.entry);
      }

      if (row.entry.stage) {
        const nextStage = row.entry.stage;
        const currentOption = nextStage.options.find((option) => option.isCurrent);

        clearDebounce();
        setPaletteState(
          enterPaletteStage(nextStage, currentOption ? `${STAGE_ENTRY_ID_PREFIX}${currentOption.id}` : null)
        );
        focusInput();
        return;
      }

      if (row.entry.keepOpen) {
        void row.entry.run();
        focusInput();
        return;
      }

      onClose();
      void row.entry.run();
    },
    [clearDebounce, enterScope, focusInput, onClose, setPaletteState]
  );

  const runSecondary = useCallback(
    (entry: PaletteEntry) => {
      if (!entry.secondary) {
        return;
      }

      if (entry.isPersistentRecent) {
        recordRecentEntry(entry);
      }
      onClose();
      void entry.secondary.run();
    },
    [onClose]
  );

  const moveActive = useCallback(
    (offset: number, scrollToIndex: ScrollToIndex) => {
      if (navigableRows.length === 0) {
        return;
      }

      const position = effectiveActive
        ? navigableRows.findIndex((candidate) => candidate.row.id === effectiveActive.row.id)
        : -1;
      const nextPosition =
        Math.abs(offset) === 1
          ? (position + offset + navigableRows.length) % navigableRows.length
          : Math.min(Math.max(position + offset, 0), navigableRows.length - 1);
      const next = navigableRows[nextPosition];

      if (!next) {
        return;
      }

      setPaletteState((current) => changePaletteSelection(current, next.row.id));
      previewRow(next.row);
      scrollToIndex(next.index);
    },
    [effectiveActive, navigableRows, previewRow, setPaletteState]
  );

  const onEscape = useCallback(() => {
    if (stage || scopeProviderKey) {
      popOverlay();
      return;
    }

    onClose();
  }, [onClose, popOverlay, scopeProviderKey, stage]);

  const onContentKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        onEscape();
      }
    },
    [onEscape]
  );

  const onSearchKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>, scrollToIndex: ScrollToIndex) => {
      if (event.key === 'ArrowDown') {
        event.preventDefault();
        moveActive(1, scrollToIndex);
      } else if (event.key === 'ArrowUp') {
        event.preventDefault();
        moveActive(-1, scrollToIndex);
      } else if (event.key === 'PageDown') {
        event.preventDefault();
        moveActive(PAGE_JUMP_ROWS, scrollToIndex);
      } else if (event.key === 'PageUp') {
        event.preventDefault();
        moveActive(-PAGE_JUMP_ROWS, scrollToIndex);
      } else if (event.key === 'Tab' && activeRow?.kind === 'scope') {
        event.preventDefault();
        enterScope(activeRow.providerKey);
      } else if (event.key === 'Backspace' && query.length === 0 && (stage || scopeProviderKey)) {
        event.preventDefault();
        popOverlay();
      } else if (event.key === 'Enter') {
        event.preventDefault();

        if (!activeRow) {
          return;
        }

        if ((event.metaKey || event.ctrlKey) && activeRow.kind === 'entry' && activeRow.entry.secondary) {
          runSecondary(activeRow.entry);
          return;
        }

        runRow(activeRow);
      }
    },
    [activeRow, enterScope, moveActive, popOverlay, query.length, runRow, runSecondary, scopeProviderKey, stage]
  );

  const onRowActive = useCallback(
    (rowId: string) => {
      const row = rows.find((candidate) => candidate.id === rowId);
      setPaletteState((current) => changePaletteSelection(current, rowId));
      previewRow(row);
    },
    [previewRow, rows, setPaletteState]
  );

  return {
    activeRow,
    activeRowId: effectiveActive?.row.id ?? null,
    onContentKeyDown,
    onRowActive,
    onRowRun: runRow,
    onSearchKeyDown,
    secondaryHint: activeRow?.kind === 'entry' ? activeRow.entry.secondary?.label : undefined,
  };
};
