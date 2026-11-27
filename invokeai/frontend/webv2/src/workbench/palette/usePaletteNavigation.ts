import type { Dispatch, KeyboardEvent, SetStateAction } from 'react';

import { createListCollection } from '@chakra-ui/react';
import { useCallback, useMemo } from 'react';

import type { NavigablePaletteRow, PaletteEntry, PaletteRow, PaletteStage } from './entries';
import type { PaletteState } from './paletteState';

import { getNavigablePaletteRows, resolveActivePaletteRow, STAGE_ENTRY_ID_PREFIX } from './entries';
import { changePaletteSelection, enterPaletteStage } from './paletteState';
import { recordRecentEntry } from './recents';

const PAGE_JUMP_ROWS = 8;

const getRowText = (row: NavigablePaletteRow): string => (row.kind === 'entry' ? row.entry.title : row.label);

// Keys that compose IME text belong to the composition; 229 is how some engines report them.
const isCompositionKey = (event: KeyboardEvent<HTMLElement>) => event.nativeEvent.isComposing || event.keyCode === 229;

const moveCaretToEdge = (input: HTMLInputElement, key: 'End' | 'Home') => {
  const position = key === 'Home' ? 0 : input.value.length;

  input.setSelectionRange(position, position);
};

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
  const navigableRows = useMemo(() => getNavigablePaletteRows(rows), [rows]);
  const collection = useMemo(
    () =>
      createListCollection({
        itemToString: getRowText,
        itemToValue: (row: NavigablePaletteRow) => row.id,
        items: navigableRows.map((candidate) => candidate.row),
      }),
    [navigableRows]
  );
  const effectiveActive = useMemo(
    () => resolveActivePaletteRow(navigableRows, activeRowId),
    [activeRowId, navigableRows]
  );
  const activeRow = effectiveActive?.row;

  const runRow = useCallback(
    (row: NavigablePaletteRow) => {
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

  /** Enter or a click on a row runs its primary action, whatever modifiers are held. */
  const onSelectRow = useCallback(
    (rowId: string) => {
      const row = navigableRows.find((candidate) => candidate.row.id === rowId)?.row;

      if (row) {
        runRow(row);
      }
    },
    [navigableRows, runRow]
  );

  const onRowActive = useCallback(
    (rowId: string) => {
      if (rowId === effectiveActive?.row.id) {
        return;
      }

      const row = rows.find((candidate) => candidate.id === rowId);
      setPaletteState((current) => changePaletteSelection(current, rowId));
      previewRow(row);
    },
    [effectiveActive, previewRow, rows, setPaletteState]
  );

  const jumpActive = useCallback(
    (offset: number) => {
      const position = effectiveActive
        ? navigableRows.findIndex((candidate) => candidate.row.id === effectiveActive.row.id)
        : -1;
      const next = navigableRows[Math.min(Math.max(position + offset, 0), navigableRows.length - 1)];

      if (next) {
        onRowActive(next.row.id);
      }
    },
    [effectiveActive, navigableRows, onRowActive]
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
      if (event.key !== 'Escape') {
        return;
      }
      // An Escape that cancels IME composition stays inside the palette but neither pops nor closes it.
      event.stopPropagation();
      if (!isCompositionKey(event)) {
        event.preventDefault();
        onEscape();
      }
    },
    [onEscape]
  );

  /**
   * Keys the combobox does not own; it handles arrows, plain Enter and pointer highlight. It runs after this handler
   * and skips prevented events, so Home/End, with Alt too, are prevented here and move the caret by hand: the combobox
   * would move the highlight instead. It leaves Ctrl, ⌘ and Shift with Home/End to the input. Mod+Enter is the
   * keyboard's only way to a secondary action; a modified click stays primary.
   */
  const onSearchKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      if (isCompositionKey(event)) {
        return;
      }

      if (
        event.key === 'Enter' &&
        (event.metaKey || event.ctrlKey) &&
        activeRow?.kind === 'entry' &&
        activeRow.entry.secondary
      ) {
        event.preventDefault();
        runSecondary(activeRow.entry);
      } else if (event.key === 'PageDown' || event.key === 'PageUp') {
        event.preventDefault();
        jumpActive(event.key === 'PageDown' ? PAGE_JUMP_ROWS : -PAGE_JUMP_ROWS);
      } else if ((event.key === 'Home' || event.key === 'End') && !event.ctrlKey && !event.metaKey && !event.shiftKey) {
        event.preventDefault();
        moveCaretToEdge(event.currentTarget, event.key);
      } else if (event.key === 'Tab' && activeRow?.kind === 'scope') {
        event.preventDefault();
        enterScope(activeRow.providerKey);
      } else if (event.key === 'Backspace' && query.length === 0 && (stage || scopeProviderKey)) {
        event.preventDefault();
        popOverlay();
      }
    },
    [activeRow, enterScope, jumpActive, popOverlay, query.length, runSecondary, scopeProviderKey, stage]
  );

  return {
    activeRow,
    activeRowId: effectiveActive?.row.id ?? null,
    collection,
    onContentKeyDown,
    onRowActive,
    onSearchKeyDown,
    onSelectRow,
    secondaryHint: activeRow?.kind === 'entry' ? activeRow.entry.secondary?.label : undefined,
  };
};
