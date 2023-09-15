import { Flex } from '@chakra-ui/react';
import { skipToken } from '@reduxjs/toolkit/dist/query';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import {
  TypesafeDraggableData,
  TypesafeDroppableData,
} from 'features/dnd/types';
import { memo, useMemo } from 'react';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

const ParamIPAdapterImage = () => {
  const ipAdapterInfo = useAppSelector(
    (state: RootState) => state.controlNet.ipAdapterInfo
  );

  const isIPAdapterEnabled = useAppSelector(
    (state: RootState) => state.controlNet.isIPAdapterEnabled
  );

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
        isUploadDisabled={!isIPAdapterEnabled}
        fitContainer
        // dropLabel="Set as Initial Image"
        noContentFallback={
          <IAINoContentFallback label="No IP Adapter Image Selected" />
        }
      />
    </Flex>
  );
};

export default memo(ParamIPAdapterImage);
