import type { GalleryItem, GalleryItemRef } from '@features/gallery/core/items';
import type { GalleryNavigationDirection, GalleryNavigationEntry } from '@features/gallery/core/selection';

import { toGalleryItemRef } from '@features/gallery/core/items';
import { getGalleryNavigationStep } from '@features/gallery/core/selection';
import { useEffect, useEffectEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { useGalleryUi } from './GalleryUiContext';
import { useGallerySelectionStarred, useGalleryWidget } from './GalleryWidgetContext';

const GALLERY_HOTKEYS = [
  ['gallery.selectAllOnPage', 'widgets.gallery.commands.selectAllOnPage', null, ['mod+a']],
  ['gallery.clearSelection', 'widgets.gallery.commands.clearSelection', null, ['esc']],
  ['gallery.galleryNavUp', 'widgets.gallery.commands.navigationUp', 'up', ['arrowup']],
  ['gallery.galleryNavRight', 'widgets.gallery.commands.navigationRight', 'right', ['arrowright']],
  ['gallery.galleryNavDown', 'widgets.gallery.commands.navigationDown', 'down', ['arrowdown']],
  ['gallery.galleryNavLeft', 'widgets.gallery.commands.navigationLeft', 'left', ['arrowleft']],
  ['gallery.galleryNavUpAlt', 'widgets.gallery.commands.navigationUp', 'up', ['alt+arrowup']],
  ['gallery.galleryNavRightAlt', 'widgets.gallery.commands.navigationRight', 'right', ['alt+arrowright']],
  ['gallery.galleryNavDownAlt', 'widgets.gallery.commands.navigationDown', 'down', ['alt+arrowdown']],
  ['gallery.galleryNavLeftAlt', 'widgets.gallery.commands.navigationLeft', 'left', ['alt+arrowleft']],
  ['gallery.deleteSelection', 'widgets.gallery.commands.deleteSelection', null, ['delete', 'backspace']],
  ['gallery.starImage', 'widgets.gallery.commands.toggleStarImage', null, ['.']],
  ['gallery.toggleStarredOnly', 'widgets.gallery.commands.toggleStarredOnly', null, []],
] as const satisfies readonly (readonly [string, string, GalleryNavigationDirection | null, readonly string[]])[];

/**
 * Read current handler state without re-registering commands on selection changes, avoiding palette and hotkey
 * churn.
 */
export const useGalleryGridHotkeys = ({
  actionSelectionRefs,
  columnCount,
  cursorKey,
  loadedItems,
  navigationSections,
  navigateToUnloadedSlot,
  getSelectionPage,
  scrollToEntry,
}: {
  actionSelectionRefs: GalleryItemRef[];
  columnCount: number;
  /** Where the arrow keys step from: the followed session, else the selected item. */
  cursorKey: string | null;
  /** Everything on hand for star-state lookups, strip included. */
  loadedItems: readonly GalleryItem[];
  /** The arrow-key sections in visual order: the starred strip, in progress, the listing. */
  navigationSections: readonly (readonly GalleryNavigationEntry[])[];
  /** Loads a sparse absolute slot; it becomes selectable after its page hydrates. */
  navigateToUnloadedSlot?: (absoluteIndex: number) => void;
  /** The sparse page stamp for a loaded listing item. */
  getSelectionPage?: (item: GalleryItem) => number | undefined;
  scrollToEntry: (entry: GalleryNavigationEntry) => void;
}) => {
  const { t } = useTranslation();
  const { actions, gallery, itemActions, runtime } = useGalleryWidget();
  const { followProgressSession, gallery: galleryCommands } = useGalleryUi();
  const shouldStar = useGallerySelectionStarred(actionSelectionRefs, loadedItems);

  const navigate = useEffectEvent((direction: GalleryNavigationDirection) => {
    const entry = getGalleryNavigationStep(navigationSections, cursorKey, direction, columnCount);

    if (!entry) {
      return;
    }

    const unloadedSlotMatch = entry.kind === 'session' ? /^gallery-unloaded-slot:(\d+)$/.exec(entry.id) : null;

    if (unloadedSlotMatch) {
      navigateToUnloadedSlot?.(Number(unloadedSlotMatch[1]));
      return;
    }

    if (entry.kind === 'session') {
      followProgressSession(entry.id, { revealPreview: false });
    } else {
      const selectionPage = getSelectionPage?.(entry.item);

      if (selectionPage === undefined) {
        actions.selectItem(entry.item);
      } else {
        actions.selectItem(entry.item, selectionPage);
      }
    }

    scrollToEntry(entry);
  });

  const executeGalleryHotkey = useEffectEvent((commandId: string) => {
    if (commandId === 'gallery.selectAllOnPage') {
      const primaryItem = gallery.items[0];

      if (primaryItem) {
        actions.selectItemRange(gallery.items.map(toGalleryItemRef), primaryItem);
      }
      return;
    }

    if (commandId === 'gallery.clearSelection') {
      galleryCommands.clearSelection();
      return;
    }

    if (commandId === 'gallery.deleteSelection' && actionSelectionRefs.length > 0) {
      void itemActions.deleteItems(actionSelectionRefs);
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
    const disposers = GALLERY_HOTKEYS.flatMap(([id, titleKey, direction, defaultKeys]) => [
      runtime.commands.register({
        handler: () => (direction ? navigate(direction) : executeGalleryHotkey(id)),
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
