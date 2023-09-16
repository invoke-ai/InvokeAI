import { Flex } from '@chakra-ui/react';
import { skipToken } from '@reduxjs/toolkit/dist/query';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import IAIIconButton from 'common/components/IAIIconButton';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { ipAdapterImageChanged } from 'features/controlNet/store/controlNetSlice';
import {
  TypesafeDraggableData,
  TypesafeDroppableData,
} from 'features/dnd/types';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaUndo } from 'react-icons/fa';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

const ParamIPAdapterImage = () => {
  const ipAdapterInfo = useAppSelector(
    (state: RootState) => state.controlNet.ipAdapterInfo
  );

  const isIPAdapterEnabled = useAppSelector(
    (state: RootState) => state.controlNet.isIPAdapterEnabled
  );

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const { currentData: imageDTO } = useGetImageDTOQuery(
    ipAdapterInfo.adapterImage?.image_name ?? skipToken
  );

  const draggableData = useMemo<TypesafeDraggableData | undefined>(() => {
    if (imageDTO) {
      return {
        id: 'ip-adapter-image',
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO },
      };
    }
  }, [imageDTO]);

  const droppableData = useMemo<TypesafeDroppableData | undefined>(
    () => ({
      id: 'ip-adapter-image',
      actionType: 'SET_IP_ADAPTER_IMAGE',
    }),
    []
  );

  return (
    <Flex
      sx={{
        position: 'relative',
        w: 'full',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <IAIDndImage
        imageDTO={imageDTO}
        droppableData={droppableData}
        draggableData={draggableData}
        isUploadDisabled={true}
        isDropDisabled={!isIPAdapterEnabled}
        dropLabel={t('toast.setIPAdapterImage')}
        noContentFallback={
          <IAINoContentFallback
            label={t('controlnet.ipAdapterImageFallback')}
          />
        }
      />

      {ipAdapterInfo.adapterImage && (
        <IAIIconButton
          tooltip={t('controlnet.resetIPAdapterImage')}
          aria-label={t('controlnet.resetIPAdapterImage')}
          icon={<FaUndo />}
          onClick={() => dispatch(ipAdapterImageChanged(null))}
          isDisabled={!imageDTO}
          size="sm"
          sx={{
            position: 'absolute',
            top: 3,
            right: 3,
          }}
        />
      )}
    </Flex>
  );
};

export default memo(ParamIPAdapterImage);
