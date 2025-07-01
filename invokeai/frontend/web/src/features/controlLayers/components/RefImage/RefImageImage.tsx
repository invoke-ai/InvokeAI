import { Flex, IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { UploadImageIconButton } from 'common/hooks/useImageUploadButton';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useRefImageIdContext } from 'features/controlLayers/contexts/RefImageIdContext';
import { usePullBboxIntoGlobalReferenceImage } from 'features/controlLayers/hooks/saveCanvasHooks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import type { ImageWithDims } from 'features/controlLayers/store/types';
import type { setGlobalReferenceImageDndTarget, setRegionalGuidanceReferenceImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { DndImage } from 'features/dnd/DndImage';
import { DndImageIcon } from 'features/dnd/DndImageIcon';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo, useCallback, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiBoundingBoxBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import { $isConnected } from 'services/events/stores';

type Props<T extends typeof setGlobalReferenceImageDndTarget | typeof setRegionalGuidanceReferenceImageDndTarget> = {
  image: ImageWithDims | null;
  onChangeImage: (imageDTO: ImageDTO | null) => void;
  dndTarget: T;
  dndTargetData: ReturnType<T['getData']>;
};

// Styled version of PullBboxIntoRefImageIconButton that matches DndImageIcon styling
const StyledPullBboxButton = memo(() => {
  const { t } = useTranslation();
  const id = useRefImageIdContext();
  const pullBboxIntoIPAdapter = usePullBboxIntoGlobalReferenceImage(id);
  const isBusy = useCanvasIsBusy();

  const sx = {
    minW: 0,
    svg: {
      transitionProperty: 'common',
      transitionDuration: 'normal',
      fill: 'base.100',
      _hover: { fill: 'base.50' },
      filter: `drop-shadow(0px 0px 0.1rem var(--invoke-colors-base-900))
        drop-shadow(0px 0px 0.3rem var(--invoke-colors-base-900))
        drop-shadow(0px 0px 0.3rem var(--invoke-colors-base-900))`,
    },
  };

  return (
    <IconButton
      onClick={pullBboxIntoIPAdapter}
      isDisabled={isBusy}
      variant="link"
      aria-label={t('controlLayers.pullBboxIntoReferenceImage')}
      tooltip={t('controlLayers.pullBboxIntoReferenceImage')}
      icon={<PiBoundingBoxBold size={16} />}
      sx={sx}
    />
  );
});

StyledPullBboxButton.displayName = 'StyledPullBboxButton';

export const RefImageImage = memo(
  <T extends typeof setGlobalReferenceImageDndTarget | typeof setRegionalGuidanceReferenceImageDndTarget>({
    image,
    onChangeImage,
    dndTarget,
    dndTargetData,
  }: Props<T>) => {
    const { t } = useTranslation();
    const isConnected = useStore($isConnected);
    const tab = useAppSelector(selectActiveTab);
    const { currentData: imageDTO, isError } = useGetImageDTOQuery(image?.image_name ?? skipToken);
    const handleResetControlImage = useCallback(() => {
      onChangeImage(null);
    }, [onChangeImage]);

    useEffect(() => {
      if (isConnected && isError) {
        handleResetControlImage();
      }
    }, [handleResetControlImage, isError, isConnected]);

    const onUpload = useCallback(
      (imageDTO: ImageDTO) => {
        onChangeImage(imageDTO);
      },
      [onChangeImage]
    );

    return (
      <Flex position="relative" w="full" h="full" alignItems="center" data-error={!imageDTO && !image?.image_name}>
        {!imageDTO && (
          <UploadImageIconButton
            w="full"
            h="full"
            isError={!imageDTO && !image?.image_name}
            onUpload={onUpload}
            fontSize={36}
          />
        )}
        {imageDTO && (
          <>
            <DndImage imageDTO={imageDTO} borderWidth={1} borderStyle="solid" w="full" />
            <Flex position="absolute" flexDir="column" top={2} insetInlineEnd={2} gap={1}>
              <DndImageIcon
                onClick={handleResetControlImage}
                icon={<PiArrowCounterClockwiseBold size={16} />}
                tooltip={t('common.reset')}
              />
              {tab === 'canvas' && (
                <CanvasManagerProviderGate>
                  <StyledPullBboxButton />
                </CanvasManagerProviderGate>
              )}
            </Flex>
          </>
        )}
        <DndDropTarget dndTarget={dndTarget} dndTargetData={dndTargetData} label={t('gallery.drop')} />
      </Flex>
    );
  }
);

RefImageImage.displayName = 'RefImageImage';
