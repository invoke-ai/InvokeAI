import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import IAIDndImage from 'common/components/IAIDndImage';
import IAIDndImageIcon from 'common/components/IAIDndImageIcon';
import { useNanoid } from 'common/hooks/useNanoid';
import type { ImageWithDims } from 'features/controlLayers/store/types';
import type { ImageDraggableData, TypesafeDroppableData } from 'features/dnd/types';
import { memo, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { ImageDTO, PostUploadAction } from 'services/api/types';
import { $isConnected } from 'services/events/stores';

type Props = {
  image: ImageWithDims | null;
  onChangeImage: (imageDTO: ImageDTO | null) => void;
  droppableData: TypesafeDroppableData;
  postUploadAction: PostUploadAction;
};

export const IPAdapterImagePreview = memo(({ image, onChangeImage, droppableData, postUploadAction }: Props) => {
  const { t } = useTranslation();
  const isConnected = useStore($isConnected);
  const dndId = useNanoid('ip_adapter_image_preview');

  const { currentData: controlImage, isError: isErrorControlImage } = useGetImageDTOQuery(
    image?.image_name ?? skipToken
  );
  const handleResetControlImage = useCallback(() => {
    onChangeImage(null);
  }, [onChangeImage]);

  const draggableData = useMemo<ImageDraggableData | undefined>(() => {
    if (controlImage) {
      return {
        id: dndId,
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO: controlImage },
      };
    }
  }, [controlImage, dndId]);

  useEffect(() => {
    if (isConnected && isErrorControlImage) {
      handleResetControlImage();
    }
  }, [handleResetControlImage, isConnected, isErrorControlImage]);

  return (
    <Flex
      position="relative"
      w="full"
      h="full"
      alignItems="center"
      borderColor="error.500"
      borderStyle="solid"
      borderWidth={controlImage ? 0 : 1}
      borderRadius="base"
    >
      <IAIDndImage
        draggableData={draggableData}
        droppableData={droppableData}
        imageDTO={controlImage}
        postUploadAction={postUploadAction}
      />

      {controlImage && (
        <Flex position="absolute" flexDir="column" top={2} insetInlineEnd={2} gap={1}>
          <IAIDndImageIcon
            onClick={handleResetControlImage}
            icon={<PiArrowCounterClockwiseBold size={16} />}
            tooltip={t('common.reset')}
          />
        </Flex>
      )}
    </Flex>
  );
});

IPAdapterImagePreview.displayName = 'IPAdapterImagePreview';
