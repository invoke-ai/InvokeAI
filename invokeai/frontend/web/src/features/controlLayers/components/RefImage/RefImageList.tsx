import { Button, Collapse, Divider, Flex, IconButton } from '@invoke-ai/ui-library';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { RefImagePreview } from 'features/controlLayers/components/RefImage/RefImagePreview';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { RefImageIdContext } from 'features/controlLayers/contexts/RefImageIdContext';
import { getDefaultRefImageConfig } from 'features/controlLayers/hooks/addLayerHooks';
import { useNewGlobalReferenceImageFromBbox } from 'features/controlLayers/hooks/saveCanvasHooks';
import { useCanvasIsBusySafe } from 'features/controlLayers/hooks/useCanvasIsBusy';
import {
  refImageAdded,
  selectIsRefImagePanelOpen,
  selectRefImageEntityIds,
  selectSelectedRefEntityId,
} from 'features/controlLayers/store/refImagesSlice';
import { imageDTOToCroppableImage } from 'features/controlLayers/store/util';
import { addGlobalReferenceImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiBoundingBoxBold, PiUploadBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

import { RefImageHeader } from './RefImageHeader';
import { RefImageSettings } from './RefImageSettings';

export const RefImageList = memo(() => {
  const ids = useAppSelector(selectRefImageEntityIds);
  const isPanelOpen = useAppSelector(selectIsRefImagePanelOpen);
  const selectedEntityId = useAppSelector(selectSelectedRefEntityId);

  return (
    <Flex flexDir="column">
      <Flex gap={2} h={16}>
        {ids.map((id) => (
          <RefImageIdContext.Provider key={id} value={id}>
            <RefImagePreview />
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
