import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { Flex, Image } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import type { AppDispatch, AppGetState } from 'app/store/store';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { uniq } from 'es-toolkit';
import { singleCanvasProjectDndSource } from 'features/dnd/dnd';
import { firefoxDndFix } from 'features/dnd/util';
import { useCanvasProjectContextMenu } from 'features/gallery/components/ContextMenu/CanvasProjectContextMenu';
import { selectGallerySlice, selectionChanged } from 'features/gallery/store/gallerySlice';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { VIEWER_PANEL_ID } from 'features/ui/layouts/shared';
import type { MouseEvent, MouseEventHandler } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef } from 'react';
import { galleryApi } from 'services/api/endpoints/gallery';
import type { CanvasProjectDTO } from 'services/api/types';

import { galleryItemContainerSX } from './galleryItemContainerSX';
import { GalleryItemProjectBadge } from './GalleryItemProjectBadge';

interface Props {
  projectDTO: CanvasProjectDTO;
}

/**
 * Returns the ordered name list from the most recently cached polymorphic gallery query, so
 * shift-range selection can span across images/videos/projects.
 */
const selectCachedGalleryItemNames = (state: ReturnType<AppGetState>): string[] => {
  const entries = galleryApi.util.selectInvalidatedBy(state, ['GalleryItemNameList']);
  for (const entry of entries) {
    if (entry.endpointName !== 'getGalleryItemNames') {
      continue;
    }
    const data = galleryApi.endpoints.getGalleryItemNames.select(entry.originalArgs)(state).data;
    if (data) {
      return data.items.map((ref) => ref.name);
    }
  }
  return [];
};

/**
 * Click handler mirroring the image/video grid behavior. Canvas projects don't participate in
 * alt-click comparison (comparison is image-only), so alt-click degrades to a plain selection.
 */
const buildOnClick =
  (projectName: string, dispatch: AppDispatch, getState: AppGetState) => (e: MouseEvent<HTMLDivElement>) => {
    const { shiftKey, ctrlKey, metaKey, altKey } = e;
    const state = getState();
    const itemNames = selectCachedGalleryItemNames(state);

    if (itemNames.length === 0) {
      if (!shiftKey && !ctrlKey && !metaKey && !altKey) {
        dispatch(selectionChanged([projectName]));
      }
      return;
    }

    const selection = state.gallery.selection;

    if (altKey) {
      dispatch(selectionChanged([projectName]));
    } else if (shiftKey) {
      const lastSelectedItem = selection.at(-1);
      const lastClickedIndex = itemNames.findIndex((name) => name === lastSelectedItem);
      const currentClickedIndex = itemNames.findIndex((name) => name === projectName);
      if (lastClickedIndex > -1 && currentClickedIndex > -1) {
        const start = Math.min(lastClickedIndex, currentClickedIndex);
        const end = Math.max(lastClickedIndex, currentClickedIndex);
        const itemsToSelect = itemNames.slice(start, end + 1);
        if (currentClickedIndex < lastClickedIndex) {
          itemsToSelect.reverse();
        }
        dispatch(selectionChanged(uniq(selection.concat(itemsToSelect))));
      }
    } else if (ctrlKey || metaKey) {
      if (selection.some((n) => n === projectName) && selection.length > 1) {
        dispatch(selectionChanged(uniq(selection.filter((n) => n !== projectName))));
      } else {
        dispatch(selectionChanged(uniq(selection.concat(projectName))));
      }
    } else {
      dispatch(selectionChanged([projectName]));
    }
  };

export const GalleryCanvasProjectItem = memo(({ projectDTO }: Props) => {
  const store = useAppStore();
  const ref = useRef<HTMLDivElement>(null);

  const selectIsSelected = useMemo(
    () => createSelector(selectGallerySlice, (gallery) => gallery.selection.some((n) => n === projectDTO.project_name)),
    [projectDTO.project_name]
  );
  const isSelected = useAppSelector(selectIsSelected);

  const onClick = useMemo(
    () => buildOnClick(projectDTO.project_name, store.dispatch, store.getState),
    [projectDTO, store]
  );

  const onDoubleClick = useCallback<MouseEventHandler<HTMLDivElement>>(() => {
    navigationApi.focusPanelInActiveTab(VIEWER_PANEL_ID);
  }, []);

  // Right-click / long-press context menu (delete, download, load).
  useCanvasProjectContextMenu(projectDTO, ref);

  // Drag source: drop the project onto a board to assign it. Mirrors GalleryVideoItem.
  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }
    return combine(
      firefoxDndFix(element),
      draggable({
        element,
        getInitialData: () => singleCanvasProjectDndSource.getData({ projectDTO }, projectDTO.project_name),
      })
    );
  }, [projectDTO]);

  return (
    <Flex
      ref={ref}
      sx={galleryItemContainerSX}
      data-item-id={projectDTO.project_name}
      role="button"
      onClick={onClick}
      onDoubleClick={onDoubleClick}
      data-selected={isSelected}
    >
      {projectDTO.thumbnail_url ? (
        <Image
          pointerEvents="none"
          src={projectDTO.thumbnail_url}
          objectFit="contain"
          maxW="full"
          maxH="full"
          borderRadius="base"
        />
      ) : (
        <Flex w="full" h="full" alignItems="center" justifyContent="center" bg="base.800" borderRadius="base" />
      )}
      <GalleryItemProjectBadge />
    </Flex>
  );
});

GalleryCanvasProjectItem.displayName = 'GalleryCanvasProjectItem';
