import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import type { ImageWithDims } from 'features/controlLayers/store/types';
import type { Dnd } from 'features/dnd2/dnd';
import { DndDropTarget } from 'features/dnd2/DndDropTarget';
import { DndImage } from 'features/dnd2/DndImage';
import { DndImageIcon } from 'features/dnd2/DndImageIcon';
import { memo, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { ImageDTO, PostUploadAction } from 'services/api/types';
import { $isConnected } from 'services/events/stores';

const sx = {
  position: 'relative',
  w: 'full',
  h: 'full',
  alignItems: 'center',
  borderColor: 'error.500',
  borderStyle: 'solid',
  borderWidth: 0,
  borderRadius: 'base',
  '&[data-error=true]': {
    borderWidth: 1,
  },
} satisfies SystemStyleObject;

type Props = {
  image: ImageWithDims | null;
  onChangeImage: (imageDTO: ImageDTO | null) => void;
  targetData: Dnd.types['TargetDataUnion'];
  postUploadAction: PostUploadAction;
};

export const IPAdapterImagePreview = memo(({ image, onChangeImage, targetData, postUploadAction }: Props) => {
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

  return (
    <Flex sx={sx} data-error={!imageDTO && !image?.image_name}>
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
      <DndDropTarget targetData={targetData} label={t('gallery.drop')} />
    </Flex>
  );
});

IPAdapterImagePreview.displayName = 'IPAdapterImagePreview';
