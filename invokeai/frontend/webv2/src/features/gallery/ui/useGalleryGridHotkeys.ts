import type { GalleryItem, GalleryItemKey, GalleryItemRef } from '@features/gallery/core/items';
import type { GalleryNavigationDirection, GalleryNavigationEntry } from '@features/gallery/core/selection';

import { toGalleryItemKey, toGalleryItemRef } from '@features/gallery/core/items';
import { getGalleryNavigationCursor, getGalleryNavigationStep } from '@features/gallery/core/selection';
import { useEffect, useEffectEvent, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { useGalleryUi } from './GalleryUiContext';
import { useGallerySelectionStarred, useGalleryWidget } from './GalleryWidgetContext';

/**
 * How an arrow moves: `select` replaces the selection with the next tile, `extend` selects the range from the anchor
 * to it, and `focus` moves keyboard focus alone, so a toggle can then build a discontiguous selection.
 */
type GalleryNavigationMode = 'extend' | 'focus' | 'select';

const GALLERY_HOTKEYS = [
  ['gallery.selectAllOnPage', 'widgets.gallery.commands.selectAllOnPage', null, ['mod+a']],
  ['gallery.clearSelection', 'widgets.gallery.commands.clearSelection', null, ['esc']],
  ['gallery.galleryNavUp', 'widgets.gallery.commands.navigationUp', ['up', 'select'], ['arrowup']],
  ['gallery.galleryNavRight', 'widgets.gallery.commands.navigationRight', ['right', 'select'], ['arrowright']],
  ['gallery.galleryNavDown', 'widgets.gallery.commands.navigationDown', ['down', 'select'], ['arrowdown']],
  ['gallery.galleryNavLeft', 'widgets.gallery.commands.navigationLeft', ['left', 'select'], ['arrowleft']],
  ['gallery.galleryNavUpAlt', 'widgets.gallery.commands.navigationUp', ['up', 'select'], ['alt+arrowup']],
  ['gallery.galleryNavRightAlt', 'widgets.gallery.commands.navigationRight', ['right', 'select'], ['alt+arrowright']],
  ['gallery.galleryNavDownAlt', 'widgets.gallery.commands.navigationDown', ['down', 'select'], ['alt+arrowdown']],
  ['gallery.galleryNavLeftAlt', 'widgets.gallery.commands.navigationLeft', ['left', 'select'], ['alt+arrowleft']],
  ['gallery.extendSelectionUp', 'widgets.gallery.commands.extendSelectionUp', ['up', 'extend'], ['shift+arrowup']],
  [
    'gallery.extendSelectionRight',
    'widgets.gallery.commands.extendSelectionRight',
    ['right', 'extend'],
    ['shift+arrowright'],
  ],
  [
    'gallery.extendSelectionDown',
    'widgets.gallery.commands.extendSelectionDown',
    ['down', 'extend'],
    ['shift+arrowdown'],
  ],
  [
    'gallery.extendSelectionLeft',
    'widgets.gallery.commands.extendSelectionLeft',
    ['left', 'extend'],
    ['shift+arrowleft'],
  ],
  ['gallery.moveFocusUp', 'widgets.gallery.commands.moveFocusUp', ['up', 'focus'], ['mod+arrowup']],
  ['gallery.moveFocusRight', 'widgets.gallery.commands.moveFocusRight', ['right', 'focus'], ['mod+arrowright']],
  ['gallery.moveFocusDown', 'widgets.gallery.commands.moveFocusDown', ['down', 'focus'], ['mod+arrowdown']],
  ['gallery.moveFocusLeft', 'widgets.gallery.commands.moveFocusLeft', ['left', 'focus'], ['mod+arrowleft']],
  ['gallery.toggleFocusedInSelection', 'widgets.gallery.commands.toggleFocusedInSelection', null, []],
  ['gallery.deleteSelection', 'widgets.gallery.commands.deleteSelection', null, ['delete', 'backspace']],
  ['gallery.starImage', 'widgets.gallery.commands.toggleStarImage', null, ['.']],
  ['gallery.toggleStarredOnly', 'widgets.gallery.commands.toggleStarredOnly', null, []],
] as const satisfies readonly (readonly [
  string,
  string,
  readonly [GalleryNavigationDirection, GalleryNavigationMode] | null,
  readonly string[],
])[];

/**
 * Read current handler state without re-registering commands on selection changes, avoiding palette and hotkey
 * churn.
 */
export const useGalleryGridHotkeys = ({
  actionSelectionRefs,
  columnCount,
  getCursorCandidates,
  getDialogReturnFocus,
  getFirstVisibleTileKey,
  getFocusedItem,
  loadedItems,
  moveToEntry,
  navigationSections,
  navigateToUnloadedSlot,
  getSelectionPage,
  selectItemRange,
  toggleItem,
}: {
  actionSelectionRefs: GalleryItemRef[];
  columnCount: number;
  /**
   * Where the arrow keys step from, in priority order: the tile holding keyboard focus, the followed session, the
   * selected item. The first the grid shows is the cursor.
   */
  getCursorCandidates: () => (string | null)[];
  /** Where focus returns from a dialog a command opens; undefined leaves that to the dialog. */
  getDialogReturnFocus: () => (() => HTMLElement | null) | undefined;
  /** The first thumbnail in view: where the arrows land when no cursor is on screen. */
  getFirstVisibleTileKey: () => string | null;
  /** The thumbnail holding keyboard focus, if any. */
  getFocusedItem: () => GalleryItem | null;
  /** Everything on hand for star-state lookups, strip included. */
  loadedItems: readonly GalleryItem[];
  /**
   * Applies `select` (none for a focus-only move), brings the entry's tile into view, and moves keyboard focus there
   * when the grid holds it.
   */
  moveToEntry: (entry: GalleryNavigationEntry, select: (() => void) | null) => void;
  /** The arrow-key sections in visual order: the starred strip, in progress, the listing. */
  navigationSections: readonly (readonly GalleryNavigationEntry[])[];
  /** Loads a sparse absolute slot; it becomes selectable after its page hydrates. */
  navigateToUnloadedSlot?: (absoluteIndex: number) => void;
  /** The sparse page stamp for a loaded listing item. */
  getSelectionPage?: (item: GalleryItem) => number | undefined;
  selectItemRange: (
    item: GalleryItem,
    options?: { anchorKey?: GalleryItemKey | null; selectionPage?: number }
  ) => Promise<void>;
  toggleItem: (item: GalleryItem) => void;
}) => {
  const { t } = useTranslation();
  const { actions, gallery, itemActions, runtime } = useGalleryWidget();
  const { followProgressSession, gallery: galleryCommands } = useGalleryUi();
  const shouldStar = useGallerySelectionStarred(actionSelectionRefs, loadedItems);
  // A run of Shift+arrows keeps the anchor it started from; the range it last reached says whether it is still running.
  const keyboardRangeRef = useRef<{ anchorKey: GalleryItemKey | null; reachedKey: GalleryItemKey } | null>(null);

  const navigate = useEffectEvent((direction: GalleryNavigationDirection, mode: GalleryNavigationMode) => {
    const cursorKey = getGalleryNavigationCursor(navigationSections, getCursorCandidates());
    // Ranges and focus moves step between items only: in-progress sessions are followed, never selected. They stay in
    // the sections, since one can be where the step starts.
    const firstVisibleKey = cursorKey === null ? getFirstVisibleTileKey() : null;
    // With no cursor on screen the arrows start where the user is looking, not at the top of the sequence.
    const entry =
      (firstVisibleKey === null
        ? null
        : navigationSections
            .flat()
            .find((candidate) => candidate.kind === 'item' && toGalleryItemKey(candidate.item) === firstVisibleKey)) ??
      getGalleryNavigationStep(navigationSections, [cursorKey], direction, columnCount, {
        itemsOnly: mode !== 'select',
      });

    if (!entry) {
      return;
    }

    const unloadedSlotMatch = entry.kind === 'session' ? /^gallery-unloaded-slot:(\d+)$/.exec(entry.id) : null;

    if (unloadedSlotMatch) {
      navigateToUnloadedSlot?.(Number(unloadedSlotMatch[1]));
      return;
    }

    if (entry.kind === 'session') {
      moveToEntry(entry, () => followProgressSession(entry.id, { revealPreview: false }));
      return;
    }

    if (mode === 'select') {
      const selectionPage = getSelectionPage?.(entry.item);

      moveToEntry(entry, () =>
        selectionPage === undefined ? actions.selectItem(entry.item) : actions.selectItem(entry.item, selectionPage)
      );
      return;
    }

    if (mode === 'focus') {
      moveToEntry(entry, null);
      return;
    }

    const range = keyboardRangeRef.current;
    const anchorKey = range && range.reachedKey === cursorKey ? range.anchorKey : gallery.selectedItemKey;

    keyboardRangeRef.current = { anchorKey, reachedKey: toGalleryItemKey(entry.item) };
    moveToEntry(
      entry,
      () => void selectItemRange(entry.item, { anchorKey, selectionPage: getSelectionPage?.(entry.item) })
    );
  });

  const executeGalleryHotkey = useEffectEvent((commandId: string) => {
    if (commandId === 'gallery.selectAllOnPage') {
      const primaryItem = gallery.items[0];

      if (primaryItem) {
        actions.selectItemRange(gallery.items.map(toGalleryItemRef), primaryItem);
      }
      return;
    }

    if (commandId === 'gallery.toggleFocusedInSelection') {
      const item = getFocusedItem();

      if (item) {
        toggleItem(item);
      }
      return;
    }

    if (commandId === 'gallery.clearSelection') {
      galleryCommands.clearSelection();
      return;
    }

    if (commandId === 'gallery.deleteSelection' && actionSelectionRefs.length > 0) {
      void itemActions.deleteItems(actionSelectionRefs, { returnFocus: getDialogReturnFocus() });
      return;
    }

    if (commandId === 'gallery.starImage' && actionSelectionRefs.length > 0) {
      void itemActions.setItemsStarred(actionSelectionRefs, shouldStar);
      return;
    }

    if (commandId === 'gallery.toggleStarredOnly' && gallery.semanticImageQuery === null) {
      actions.setStarredOnly(!gallery.starredOnly);
    }
  });

  useEffect(() => {
    const disposers = GALLERY_HOTKEYS.flatMap(([id, titleKey, navigation, defaultKeys]) => [
      runtime.commands.register({
        handler: () => (navigation ? navigate(navigation[0], navigation[1]) : executeGalleryHotkey(id)),
        id,
        title: t(titleKey),
      }),
      runtime.hotkeys.register({
        commandId: id,
        defaultKeys: [...defaultKeys],
        id,
        title: t(titleKey),
      }),
    ]);

    return () => {
      disposers.forEach((dispose) => dispose());
    };
  }, [runtime.commands, runtime.hotkeys, t]);
};
