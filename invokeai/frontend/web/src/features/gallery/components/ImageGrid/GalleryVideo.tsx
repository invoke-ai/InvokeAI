import type { FlexProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, Icon, Image } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import type { AppDispatch, AppGetState } from 'app/store/store';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { uniq } from 'es-toolkit';
import type { DndDragPreviewMultipleImageState } from 'features/dnd/DndDragPreviewMultipleImage';
import { createMultipleImageDragPreview } from 'features/dnd/DndDragPreviewMultipleImage';
import type { DndDragPreviewSingleImageState } from 'features/dnd/DndDragPreviewSingleImage';
import { createSingleImageDragPreview } from 'features/dnd/DndDragPreviewSingleImage';
import {
  selectGetImageNamesQueryArgs,
} from 'features/gallery/store/gallerySelectors';
import { imageToCompareChanged, selectGallerySlice, selectionChanged } from 'features/gallery/store/gallerySlice';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { VIEWER_PANEL_ID } from 'features/ui/layouts/shared';
import type { MouseEvent, MouseEventHandler } from 'react';
import { memo, useCallback, useMemo, useRef, useState } from 'react';
import { PiImageBold } from 'react-icons/pi';
import { imagesApi } from 'services/api/endpoints/images';
import type { VideoDTO } from 'services/api/types';

import { GalleryResourceHoverIcons } from './GalleryResourceHoverIcons';

const galleryImageContainerSX = {
  containerType: 'inline-size',
  w: 'full',
  h: 'full',
  '.gallery-image-size-badge': {
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
  videoDTO: VideoDTO;
}

const buildOnClick =
  (imageName: string, dispatch: AppDispatch, getState: AppGetState) => (e: MouseEvent<HTMLDivElement>) => {
    const { shiftKey, ctrlKey, metaKey, altKey } = e;
    const state = getState();
    const queryArgs = selectGetImageNamesQueryArgs(state);
    const imageNames = imagesApi.endpoints.getImageNames.select(queryArgs)(state).data?.image_names ?? [];

    // If we don't have the image names cached, we can't perform selection operations
    // This can happen if the user clicks on an image before the names are loaded
    if (imageNames.length === 0) {
      // For basic click without modifiers, we can still set selection
      if (!shiftKey && !ctrlKey && !metaKey && !altKey) {
        dispatch(selectionChanged([imageName]));
      }
      return;
    }

    const selection = state.gallery.selection;

    if (altKey) {
      if (state.gallery.imageToCompare === imageName) {
        dispatch(imageToCompareChanged(null));
      } else {
        dispatch(imageToCompareChanged(imageName));
      }
    } else if (shiftKey) {
      const rangeEndImageName = imageName;
      const lastSelectedImage = selection.at(-1);
      const lastClickedIndex = imageNames.findIndex((name) => name === lastSelectedImage);
      const currentClickedIndex = imageNames.findIndex((name) => name === rangeEndImageName);
      if (lastClickedIndex > -1 && currentClickedIndex > -1) {
        // We have a valid range!
        const start = Math.min(lastClickedIndex, currentClickedIndex);
        const end = Math.max(lastClickedIndex, currentClickedIndex);
        const imagesToSelect = imageNames.slice(start, end + 1);
        dispatch(selectionChanged(uniq(selection.concat(imagesToSelect))));
      }
    } else if (ctrlKey || metaKey) {
      if (selection.some((n) => n === imageName) && selection.length > 1) {
        dispatch(selectionChanged(uniq(selection.filter((n) => n !== imageName))));
      } else {
        dispatch(selectionChanged(uniq(selection.concat(imageName))));
      }
    } else {
      dispatch(selectionChanged([imageName]));
    }
  };

export const GalleryVideo = memo(({ videoDTO }: Props) => {
  const store = useAppStore();
  const [isDragging, setIsDragging] = useState(false);
  const [dragPreviewState, setDragPreviewState] = useState<
    DndDragPreviewSingleImageState | DndDragPreviewMultipleImageState | null
  >(null);
  const ref = useRef<HTMLDivElement>(null);
  const selectIsSelectedForCompare = useMemo(
    () => createSelector(selectGallerySlice, (gallery) => gallery.imageToCompare === videoDTO.video_id),
    [videoDTO.video_id]
  );
  const isSelectedForCompare = useAppSelector(selectIsSelectedForCompare);
  const selectIsSelected = useMemo(
    () => createSelector(selectGallerySlice, (gallery) => gallery.selection.includes(videoDTO.video_id)),
    [videoDTO.video_id]
  );
  const isSelected = useAppSelector(selectIsSelected);

  const [isHovered, setIsHovered] = useState(false);

  const onMouseOver = useCallback(() => {
    setIsHovered(true);
  }, []);

  const onMouseOut = useCallback(() => {
    setIsHovered(false);
  }, []);

  const onClick = useMemo(() => buildOnClick(videoDTO.video_id, store.dispatch, store.getState), [videoDTO, store]);

  const onDoubleClick = useCallback<MouseEventHandler<HTMLDivElement>>(() => {
    store.dispatch(imageToCompareChanged(null));
    navigationApi.focusPanelInActiveTab(VIEWER_PANEL_ID);
  }, [store]);

  return (
    <>
      <Flex
        ref={ref}
        sx={galleryImageContainerSX}
        data-is-dragging={isDragging}
        data-video-id={videoDTO.video_id}
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
          src="" // TODO: Add video thumbnail
          w={videoDTO.width}
          fallback={<GalleryVideoPlaceholder />}
          objectFit="contain"
          maxW="full"
          maxH="full"
          borderRadius="base"
        />
        <GalleryResourceHoverIcons resource={videoDTO} isHovered={isHovered} />
      </Flex>
      {dragPreviewState?.type === 'multiple-image' ? createMultipleImageDragPreview(dragPreviewState) : null}
      {dragPreviewState?.type === 'single-image' ? createSingleImageDragPreview(dragPreviewState) : null}
    </>
  );
});

GalleryVideo.displayName = 'GalleryVideo';

export const GalleryVideoPlaceholder = memo((props: FlexProps) => (
  <Flex w="full" h="full" bg="base.850" borderRadius="base" alignItems="center" justifyContent="center" {...props}>
    <Icon as={PiImageBold} boxSize={16} color="base.800" />
  </Flex>
));

GalleryVideoPlaceholder.displayName = 'GalleryVideoPlaceholder';
