import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable, monitorForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import type { FlexProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, Icon, Image } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import type { AppDispatch, AppGetState } from 'app/store/store';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { uniq } from 'es-toolkit';
import { multipleImageDndSource, singleImageDndSource } from 'features/dnd/dnd';
import type { DndDragPreviewMultipleImageState } from 'features/dnd/DndDragPreviewMultipleImage';
import { createMultipleImageDragPreview, setMultipleImageDragPreview } from 'features/dnd/DndDragPreviewMultipleImage';
import type { DndDragPreviewSingleImageState } from 'features/dnd/DndDragPreviewSingleImage';
import { createSingleImageDragPreview, setSingleImageDragPreview } from 'features/dnd/DndDragPreviewSingleImage';
import { firefoxDndFix } from 'features/dnd/util';
import { useImageContextMenu } from 'features/gallery/components/ImageContextMenu/ImageContextMenu';
import { GalleryResourceHoverIcons } from 'features/gallery/components/ImageGrid/GalleryResourceHoverIcons';
import {
  selectGetImageNamesQueryArgs,
  selectSelectedBoardId,
  selectSelection,
} from 'features/gallery/store/gallerySelectors';
import { imageToCompareChanged, selectGallerySlice, selectionChanged } from 'features/gallery/store/gallerySlice';
import { isImageResource, isVideoResource } from 'features/gallery/store/resourceTypes';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { VIEWER_PANEL_ID } from 'features/ui/layouts/shared';
import type { MouseEvent, MouseEventHandler } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { PiImageBold, PiVideoBold } from 'react-icons/pi';
import { imagesApi } from 'services/api/endpoints/images';
import { ImageDTO, VideoDTO } from 'services/api/types';


const galleryResourceContainerSX = {
  containerType: 'inline-size',
  w: 'full',
  h: 'full',
  '.gallery-resource-size-badge': {
    '@container (max-width: 80px)': {
      '&': { display: 'none' },
    },
  },
  '&[data-is-dragging=true]': {
    opacity: 0.3,
  },
  userSelect: 'none',
  webkitUserSelect: 'none',
  position: 'relative',
  justifyContent: 'center',
  alignItems: 'center',
  aspectRatio: '1/1',
  '::before': {
    content: '""',
    display: 'inline-block',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    pointerEvents: 'none',
    borderRadius: 'base',
  },
  '&[data-selected=true]::before': {
    boxShadow:
      'inset 0px 0px 0px 3px var(--invoke-colors-invokeBlue-500), inset 0px 0px 0px 4px var(--invoke-colors-invokeBlue-800)',
  },
  '&[data-selected-for-compare=true]::before': {
    boxShadow:
      'inset 0px 0px 0px 3px var(--invoke-colors-invokeGreen-300), inset 0px 0px 0px 4px var(--invoke-colors-invokeGreen-800)',
  },
  '&:hover::before': {
    boxShadow:
      'inset 0px 0px 0px 1px var(--invoke-colors-invokeBlue-300), inset 0px 0px 0px 2px var(--invoke-colors-invokeBlue-800)',
  },
  '&:hover[data-selected=true]::before': {
    boxShadow:
      'inset 0px 0px 0px 3px var(--invoke-colors-invokeBlue-400), inset 0px 0px 0px 4px var(--invoke-colors-invokeBlue-800)',
  },
  '&:hover[data-selected-for-compare=true]::before': {
    boxShadow:
      'inset 0px 0px 0px 3px var(--invoke-colors-invokeGreen-200), inset 0px 0px 0px 4px var(--invoke-colors-invokeGreen-800)',
  },
} satisfies SystemStyleObject;

interface Props {
  resource: ImageDTO | VideoDTO;
}

const buildOnClick =
  (resourceId: string, dispatch: AppDispatch, getState: AppGetState) => (e: MouseEvent<HTMLDivElement>) => {
    const { shiftKey, ctrlKey, metaKey, altKey } = e;
    const state = getState();
    const queryArgs = selectGetImageNamesQueryArgs(state);
    const imageNames = imagesApi.endpoints.getImageNames.select(queryArgs)(state).data?.image_names ?? [];

    // If we don't have the image names cached, we can't perform selection operations
    // This can happen if the user clicks on a resource before the names are loaded
    if (imageNames.length === 0) {
      // For basic click without modifiers, we can still set selection
      if (!shiftKey && !ctrlKey && !metaKey && !altKey) {
        dispatch(selectionChanged([resourceId]));
      }
      return;
    }

    const selection = state.gallery.selection;

    if (altKey) {
      if (state.gallery.imageToCompare === resourceId) {
        dispatch(imageToCompareChanged(null));
      } else {
        dispatch(imageToCompareChanged(resourceId));
      }
    } else if (shiftKey) {
      const rangeEndResourceId = resourceId;
      const lastSelectedResource = selection.at(-1);
      const lastClickedIndex = imageNames.findIndex((name) => name === lastSelectedResource);
      const currentClickedIndex = imageNames.findIndex((name) => name === rangeEndResourceId);
      if (lastClickedIndex > -1 && currentClickedIndex > -1) {
        // We have a valid range!
        const start = Math.min(lastClickedIndex, currentClickedIndex);
        const end = Math.max(lastClickedIndex, currentClickedIndex);
        const resourcesToSelect = imageNames.slice(start, end + 1);
        dispatch(selectionChanged(uniq(selection.concat(resourcesToSelect))));
      }
    } else if (ctrlKey || metaKey) {
      if (selection.some((n) => n === resourceId) && selection.length > 1) {
        dispatch(selectionChanged(uniq(selection.filter((n) => n !== resourceId))));
      } else {
        dispatch(selectionChanged(uniq(selection.concat(resourceId))));
      }
    } else {
      dispatch(selectionChanged([resourceId]));
    }
  };

export const GalleryResource = memo(({ resource }: Props) => {
  const store = useAppStore();
  const [isDragging, setIsDragging] = useState(false);
  const [dragPreviewState, setDragPreviewState] = useState<
    DndDragPreviewSingleImageState | DndDragPreviewMultipleImageState | null
  >(null);
  const ref = useRef<HTMLDivElement>(null);
  const resourceId = useMemo(() => {
    if (isVideoResource(resource)) {
      return resource.video_id ?? '';
    } else {
      return resource.image_name;
    }
  }, [resource]);
  const selectIsSelectedForCompare = useMemo(
    () => createSelector(selectGallerySlice, (gallery) => gallery.imageToCompare === resourceId),
    [resourceId]
  );
  const isSelectedForCompare = useAppSelector(selectIsSelectedForCompare);
  const selectIsSelected = useMemo(
    () => createSelector(selectGallerySlice, (gallery) => gallery.selection.includes(resourceId)),
    [resourceId]
  );
  const isSelected = useAppSelector(selectIsSelected);

  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }
    return combine(
      firefoxDndFix(element),
      draggable({
        element,
        getInitialData: () => {
          const selection = selectSelection(store.getState());
          const boardId = selectSelectedBoardId(store.getState());
          // When we have multiple resources selected, and the dragged resource is part of the selection, initiate a
          // multi-resource drag.
          if (selection.length > 1 && selection.includes(resourceId)) {
            return multipleImageDndSource.getData({
              image_names: selection,
              board_id: boardId,
            });
          }

          // Otherwise, initiate a single-resource drag
          // For now, we'll treat videos like images for DnD
          if (isImageResource(resource)) {
            return singleImageDndSource.getData({ imageDTO: { image_name: resourceId } as any }, resourceId);
          } else {
            // For videos, we'll adapt the image DnD for now
            return singleImageDndSource.getData({ imageDTO: { image_name: resourceId } as any }, resourceId);
          }
        },
        // This is a "local" drag start event, meaning that it is only called when this specific resource is dragged.
        onDragStart: ({ source }) => {
          // When we start dragging a single resource, set the dragging state to true. This is only called when this
          // specific resource is dragged.
          if (singleImageDndSource.typeGuard(source.data)) {
            setIsDragging(true);
            return;
          }
        },
        onGenerateDragPreview: (args) => {
          if (multipleImageDndSource.typeGuard(args.source.data)) {
            setMultipleImageDragPreview({
              multipleImageDndData: args.source.data,
              onGenerateDragPreviewArgs: args,
              setDragPreviewState,
            });
          } else if (singleImageDndSource.typeGuard(args.source.data)) {
            setSingleImageDragPreview({
              singleImageDndData: args.source.data,
              onGenerateDragPreviewArgs: args,
              setDragPreviewState,
            });
          }
        },
      }),
      monitorForElements({
        // This is a "global" drag start event, meaning that it is called for all drag events.
        onDragStart: ({ source }) => {
          // When we start dragging multiple resources, set the dragging state to true if the dragged resource is part of the
          // selection. This is called for all drag events.
          if (
            multipleImageDndSource.typeGuard(source.data) &&
            source.data.payload.image_names.includes(resourceId)
          ) {
            setIsDragging(true);
          }
        },
        onDrop: () => {
          // Always set the dragging state to false when a drop event occurs.
          setIsDragging(false);
        },
      })
    );
  }, [resource, store]);

  const [isHovered, setIsHovered] = useState(false);

  const onMouseOver = useCallback(() => {
    setIsHovered(true);
  }, []);

  const onMouseOut = useCallback(() => {
    setIsHovered(false);
  }, []);

  const onClick = useMemo(() => buildOnClick(resourceId, store.dispatch, store.getState), [resourceId, store]);

  const onDoubleClick = useCallback<MouseEventHandler<HTMLDivElement>>(() => {
    store.dispatch(imageToCompareChanged(null));
    navigationApi.focusPanelInActiveTab(VIEWER_PANEL_ID);
  }, [store]);

  // For now, we'll use image context menu for both images and videos
  // Later, we can create a video-specific context menu
  if (isImageResource(resource)) {
    useImageContextMenu({ image_name: resourceId } as any, ref);
  }

  return (
    <>
      <Flex
        ref={ref}
        sx={galleryResourceContainerSX}
        data-is-dragging={isDragging}
        data-resource-id={resourceId}
        data-resource-type={isImageResource(resource) ? 'image' : 'video'}
        role="button"
        onMouseOver={onMouseOver}
        onMouseOut={onMouseOut}
        onClick={onClick}
        onDoubleClick={onDoubleClick}
        data-selected={isSelected}
        data-selected-for-compare={isSelectedForCompare}
      >
        <Image
          pointerEvents="none"
          src={isImageResource(resource) ? resource.thumbnail_url : ""}
          w={resource.width}
          fallback={<GalleryResourcePlaceholder type={isImageResource(resource) ? 'image' : 'video'} />}
          objectFit="contain"
          maxW="full"
          maxH="full"
          borderRadius="base"
        />
        <GalleryResourceHoverIcons resource={resource} isHovered={isHovered} />
      </Flex>
      {dragPreviewState?.type === 'multiple-image' ? createMultipleImageDragPreview(dragPreviewState) : null}
      {dragPreviewState?.type === 'single-image' ? createSingleImageDragPreview(dragPreviewState) : null}
    </>
  );
});

GalleryResource.displayName = 'GalleryResource';

export const GalleryResourcePlaceholder = memo((props: FlexProps & { type: 'image' | 'video' }) => (
  <Flex w="full" h="full" bg="base.850" borderRadius="base" alignItems="center" justifyContent="center" {...props}>
    <Icon as={props.type === 'video' ? PiVideoBold : PiImageBold} boxSize={16} color="base.800" />
  </Flex>
));

GalleryResourcePlaceholder.displayName = 'GalleryResourcePlaceholder';

