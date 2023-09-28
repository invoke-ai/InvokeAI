import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/dist/query';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIDndImage from 'common/components/IAIDndImage';
import IAIDndImageIcon from 'common/components/IAIDndImageIcon';
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
import { PostUploadAction } from 'services/api/types';

const selector = createSelector(
  stateSelector,
  ({ controlNet }) => {
    const { ipAdapterInfo } = controlNet;
    return { ipAdapterInfo };
  },
  defaultSelectorOptions
);

const ParamIPAdapterImage = () => {
  const { ipAdapterInfo } = useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const { currentData: imageDTO } = useGetImageDTOQuery(
    ipAdapterInfo.adapterImage ?? skipToken
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

  const postUploadAction = useMemo<PostUploadAction>(
    () => ({
      type: 'SET_IP_ADAPTER_IMAGE',
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
        postUploadAction={postUploadAction}
        dropLabel={t('toast.setIPAdapterImage')}
        noContentFallback={
          <IAINoContentFallback
            label={t('controlnet.ipAdapterImageFallback')}
          />
        }
      />

      <IAIDndImageIcon
        onClick={() => dispatch(ipAdapterImageChanged(null))}
        icon={ipAdapterInfo.adapterImage ? <FaUndo /> : undefined}
        tooltip={t('controlnet.resetIPAdapterImage')}
      />
    </Flex>
  );
};

export default memo(ParamIPAdapterImage);
