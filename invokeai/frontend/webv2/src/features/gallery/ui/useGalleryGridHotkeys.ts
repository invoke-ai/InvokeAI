import type { GalleryItemRef } from '@features/gallery/core/items';

import { shouldStarSelection, toGalleryItemKey, toGalleryItemRef } from '@features/gallery/core/items';
import { useEffect, useEffectEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { useGalleryUi } from './GalleryUiContext';
import { useGalleryWidget } from './GalleryWidgetContext';

type GalleryNavDirection = 'down' | 'left' | 'right' | 'up';

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
] as const satisfies readonly (readonly [string, string, GalleryNavDirection | null, readonly string[]])[];

/**
 * Registers the grid's commands and their default keys.
 *
 * Handlers run through `useEffectEvent` so registration depends only on the
 * runtime and the translator: re-registering every command on each selection
 * change would churn the palette and the hotkey map on every click.
 */
export const useGalleryGridHotkeys = ({
  actionSelectionRefs,
  columnCount,
  scrollToItemIndex,
}: {
  actionSelectionRefs: GalleryItemRef[];
  columnCount: number;
  scrollToItemIndex: (itemIndex: number) => void;
}) => {
  const { t } = useTranslation();
  const { actions, gallery, itemActions, runtime } = useGalleryWidget();
  const { widgets } = useGalleryUi();

  const navigate = useEffectEvent((direction: GalleryNavDirection) => {
    const itemCount = gallery.items.length;

    if (itemCount === 0) {
      return;
    }

    const selectedIndex = gallery.items.findIndex((item) => toGalleryItemKey(item) === gallery.selectedItemKey);
    const fallbackIndex = selectedIndex === -1 ? 0 : selectedIndex;
    const delta =
      direction === 'right' ? 1 : direction === 'left' ? -1 : direction === 'down' ? columnCount : -columnCount;
    const nextIndex = Math.min(itemCount - 1, Math.max(0, fallbackIndex + delta));
    const nextItem = gallery.items[nextIndex];

    if (nextItem && (nextIndex !== selectedIndex || selectedIndex === -1)) {
      actions.selectItem(nextItem);
      scrollToItemIndex(nextIndex);
    }
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
      widgets.patchGalleryValues({
        selectedImage: null,
        selectedImageName: null,
        selectedImageNames: [],
      });
      return;
    }

    if (commandId === 'gallery.deleteSelection' && actionSelectionRefs.length > 0) {
      void itemActions.deleteItems(actionSelectionRefs);
      return;
    }

    if (commandId === 'gallery.starImage' && actionSelectionRefs.length > 0) {
      void itemActions.setItemsStarred(actionSelectionRefs, shouldStarSelection(gallery.items, actionSelectionRefs));
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
