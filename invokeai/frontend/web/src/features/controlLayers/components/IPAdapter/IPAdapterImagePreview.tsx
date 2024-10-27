import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import IAIDndImageIcon from 'common/components/IAIDndImageIcon';
import type { ImageWithDims } from 'features/controlLayers/store/types';
import { DndDropTarget } from 'features/dnd2/DndDropTarget';
import { DndImage } from 'features/dnd2/DndImage';
import type { DndTargetData } from 'features/dnd2/types';
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
  targetData: DndTargetData;
  postUploadAction: PostUploadAction;
};

export const IPAdapterImagePreview = memo(({ image, onChangeImage, targetData, postUploadAction }: Props) => {
  const { t } = useTranslation();
  const isConnected = useStore($isConnected);
  const imageDTOQueryResult = useGetImageDTOQuery(image?.image_name ?? skipToken);
  const handleResetControlImage = useCallback(() => {
    onChangeImage(null);
  }, [onChangeImage]);

  useEffect(() => {
    if (isConnected && imageDTOQueryResult.isError) {
      handleResetControlImage();
    }
  }, [handleResetControlImage, imageDTOQueryResult.isError, isConnected]);

  return (
    <Flex sx={sx} data-error={!imageDTOQueryResult.currentData && !image?.image_name}>
      {imageDTOQueryResult.currentData && (
        <>
          <DndImage dndId={targetData.dndId} imageDTO={imageDTOQueryResult.currentData} />
          <Flex position="absolute" flexDir="column" top={2} insetInlineEnd={2} gap={1}>
            <IAIDndImageIcon
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
