import { monitorForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { extractClosestEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/closest-edge';
import { reorderWithEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/util/reorder-with-edge';
import { Box, Button, Collapse, Divider, Flex, IconButton } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { useAppDispatch, useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { RefImagePreview } from 'features/controlLayers/components/RefImage/RefImagePreview';
import { useRefImageListDnd } from 'features/controlLayers/components/RefImage/useRefImageListDnd';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { RefImageIdContext, useRefImageIdContext } from 'features/controlLayers/contexts/RefImageIdContext';
import { getDefaultRefImageConfig } from 'features/controlLayers/hooks/addLayerHooks';
import { useNewGlobalReferenceImageFromBbox } from 'features/controlLayers/hooks/saveCanvasHooks';
import { useCanvasIsBusySafe } from 'features/controlLayers/hooks/useCanvasIsBusy';
import {
  refImageAdded,
  refImagesReordered,
  selectIsRefImagePanelOpen,
  selectRefImageEntityIds,
  selectSelectedRefEntityId,
} from 'features/controlLayers/store/refImagesSlice';
import { imageDTOToCroppableImage } from 'features/controlLayers/store/util';
import { addGlobalReferenceImageDndTarget, singleRefImageDndSource } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { DndListDropIndicator } from 'features/dnd/DndListDropIndicator';
import { triggerPostMoveFlash } from 'features/dnd/util';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo, useEffect, useMemo, useRef } from 'react';
import { flushSync } from 'react-dom';
import { useTranslation } from 'react-i18next';
import { PiBoundingBoxBold, PiUploadBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

import { RefImageHeader } from './RefImageHeader';
import { RefImageSettings } from './RefImageSettings';

export const RefImageList = memo(() => {
  const dispatch = useAppDispatch();
  const ids = useAppSelector(selectRefImageEntityIds);
  const isPanelOpen = useAppSelector(selectIsRefImagePanelOpen);
  const selectedEntityId = useAppSelector(selectSelectedRefEntityId);

  useEffect(() => {
    return monitorForElements({
      canMonitor({ source }) {
        return singleRefImageDndSource.typeGuard(source.data);
      },
      onDrop({ location, source }) {
        const target = location.current.dropTargets[0];
        if (!target) {
          return;
        }

        const sourceData = source.data;
        const targetData = target.data;

        if (!singleRefImageDndSource.typeGuard(sourceData) || !singleRefImageDndSource.typeGuard(targetData)) {
          return;
        }

        const indexOfSource = ids.indexOf(sourceData.payload.id);
        const indexOfTarget = ids.indexOf(targetData.payload.id);

        if (indexOfTarget < 0 || indexOfSource < 0) {
          return;
        }

        if (indexOfSource === indexOfTarget) {
          return;
        }

        const closestEdgeOfTarget = extractClosestEdge(targetData);

        let edgeIndexDelta = 0;
        if (closestEdgeOfTarget === 'right') {
          edgeIndexDelta = 1;
        } else if (closestEdgeOfTarget === 'left') {
          edgeIndexDelta = -1;
        }

        if (indexOfSource === indexOfTarget + edgeIndexDelta) {
          return;
        }

        const reorderedIds = reorderWithEdge({
          list: ids,
          startIndex: indexOfSource,
          indexOfTarget,
          closestEdgeOfTarget,
          axis: 'horizontal',
        });

        flushSync(() => {
          dispatch(refImagesReordered({ ids: reorderedIds }));
        });

        logger('dnd').info({ reorderedIds }, 'Ref images reordered');

        const element = document.querySelector(`[data-ref-image-id="${sourceData.payload.id}"]`);
        if (element instanceof HTMLElement) {
          triggerPostMoveFlash(element, colorTokenToCssVar('base.700'));
        }
      },
    });
  }, [dispatch, ids]);

  return (
    <Flex flexDir="column">
      <Flex gap={2} h={16}>
        {ids.map((id) => (
          <RefImageIdContext.Provider key={id} value={id}>
            <RefImageListItem />
          </RefImageIdContext.Provider>
        ))}
        {ids.length < 5 && <AddRefImageDropTargetAndButton />}
        {ids.length >= 5 && <MaxRefImages />}
      </Flex>
      <Collapse in={isPanelOpen}>
        <Flex pt={2} w="full">
          {selectedEntityId !== null && (
            <RefImageIdContext.Provider value={selectedEntityId}>
              <Flex flexDir="column" gap={2} w="full" h="full" borderRadius="base" bg="base.800" p={2}>
                <RefImageHeader />
                <Divider />
                <RefImageSettings />
              </Flex>
            </RefImageIdContext.Provider>
          )}
        </Flex>
      </Collapse>
    </Flex>
  );
});

RefImageList.displayName = 'RefImageList';

const RefImageListItem = memo(() => {
  const id = useRefImageIdContext();
  const ref = useRef<HTMLDivElement>(null);
  const [dndListState, isDragging] = useRefImageListDnd(ref, id);

  return (
    <Box ref={ref} position="relative" h="full" data-ref-image-id={id} opacity={isDragging ? 0.3 : 1} cursor="grab">
      <RefImagePreview />
      <DndListDropIndicator dndState={dndListState} gap="var(--invoke-space-2)" />
    </Box>
  );
});
RefImageListItem.displayName = 'RefImageListItem';

const dndTargetData = addGlobalReferenceImageDndTarget.getData();

const MaxRefImages = memo(() => {
  const { t } = useTranslation();
  return (
    <Button
      position="relative"
      size="sm"
      variant="ghost"
      h="full"
      w="full"
      borderWidth="2px !important"
      borderStyle="dashed !important"
      borderRadius="base"
      isDisabled
    >
      {t('controlLayers.maxRefImages')}
    </Button>
  );
});
MaxRefImages.displayName = 'MaxRefImages';

const AddRefImageDropTargetAndButton = memo(() => {
  const { dispatch, getState } = useAppStore();
  const { t } = useTranslation();
  const tab = useAppSelector(selectActiveTab);

  const uploadOptions = useMemo(
    () =>
      ({
        onUpload: (imageDTO: ImageDTO) => {
          const config = getDefaultRefImageConfig(getState);
          config.image = imageDTOToCroppableImage(imageDTO);
          dispatch(refImageAdded({ overrides: { config } }));
        },
        allowMultiple: false,
      }) as const,
    [dispatch, getState]
  );

  const uploadApi = useImageUploadButton(uploadOptions);

  return (
    <Flex gap={1} h="full" w="full">
      <Button
        position="relative"
        size="sm"
        variant="ghost"
        h="full"
        w="full"
        borderWidth="2px !important"
        borderStyle="dashed !important"
        borderRadius="base"
        leftIcon={<PiUploadBold />}
        {...uploadApi.getUploadButtonProps()}
      >
        {t('controlLayers.referenceImage')}
        <input {...uploadApi.getUploadInputProps()} />
        <DndDropTarget label="Drop" dndTarget={addGlobalReferenceImageDndTarget} dndTargetData={dndTargetData} />
      </Button>
      {tab === 'canvas' && (
        <CanvasManagerProviderGate>
          <BboxButton />
        </CanvasManagerProviderGate>
      )}
    </Flex>
  );
});
AddRefImageDropTargetAndButton.displayName = 'AddRefImageDropTargetAndButton';

const BboxButton = memo(() => {
  const { t } = useTranslation();
  const isBusy = useCanvasIsBusySafe();
  const newGlobalReferenceImageFromBbox = useNewGlobalReferenceImageFromBbox();

  return (
    <IconButton
      size="lg"
      variant="outline"
      h="full"
      icon={<PiBoundingBoxBold />}
      onClick={newGlobalReferenceImageFromBbox}
      isDisabled={isBusy}
      aria-label={t('controlLayers.pullBboxIntoReferenceImage')}
      tooltip={t('controlLayers.pullBboxIntoReferenceImage')}
    />
  );
});
BboxButton.displayName = 'BboxButton';
