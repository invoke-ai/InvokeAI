import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import { UploadImageButton } from 'common/hooks/useImageUploadButton';
import type { ImageWithDims } from 'features/controlLayers/store/types';
import type { setGlobalReferenceImageDndTarget, setRegionalGuidanceReferenceImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { DndImage } from 'features/dnd/DndImage';
import { DndImageIcon } from 'features/dnd/DndImageIcon';
import { memo, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import { $isConnected } from 'services/events/stores';

type Props<T extends typeof setGlobalReferenceImageDndTarget | typeof setRegionalGuidanceReferenceImageDndTarget> = {
  image: ImageWithDims | null;
  onChangeImage: (imageDTO: ImageDTO | null) => void;
  dndTarget: T;
  dndTargetData: ReturnType<T['getData']>;
};

export const IPAdapterImagePreview = memo(
  <T extends typeof setGlobalReferenceImageDndTarget | typeof setRegionalGuidanceReferenceImageDndTarget>({
    image,
    onChangeImage,
    dndTarget,
    dndTargetData,
  }: Props<T>) => {
    const { t } = useTranslation();
    const isConnected = useStore($isConnected);
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
          <UploadImageButton
            w="full"
            h="full"
            isError={!imageDTO && !image?.image_name}
            onUpload={onUpload}
            fontSize={36}
          />
        )}
        {imageDTO && (
          <>
            <DndImage imageDTO={imageDTO} />
            <Flex position="absolute" flexDir="column" top={2} insetInlineEnd={2} gap={1}>
              <DndImageIcon
                onClick={handleResetControlImage}
                icon={<PiArrowCounterClockwiseBold size={16} />}
                tooltip={t('common.reset')}
              />
            </Flex>
          </>
        )}
        <DndDropTarget dndTarget={dndTarget} dndTargetData={dndTargetData} label={t('gallery.drop')} />
      </Flex>
    );
  }
);

IPAdapterImagePreview.displayName = 'IPAdapterImagePreview';
