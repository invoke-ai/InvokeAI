import { Flex, Text } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import IAIDndImageIcon from 'common/components/IAIDndImageIcon';
import type { TypesafeDraggableData, TypesafeDroppableData } from 'features/dnd/types';
import { fieldImageValueChanged } from 'features/nodes/store/nodesSlice';
import type { ImageFieldInputInstance, ImageFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { PostUploadAction } from 'services/api/types';

import type { FieldComponentProps } from './types';

const ImageFieldInputComponent = (props: FieldComponentProps<ImageFieldInputInstance, ImageFieldInputTemplate>) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const isConnected = useAppSelector((s) => s.system.isConnected);
  const { currentData: imageDTO, isError } = useGetImageDTOQuery(field.value?.image_name ?? skipToken);

  const handleReset = useCallback(() => {
    dispatch(
      fieldImageValueChanged({
        nodeId,
        fieldName: field.name,
        value: undefined,
      })
    );
  }, [dispatch, field.name, nodeId]);

  const draggableData = useMemo<TypesafeDraggableData | undefined>(() => {
    if (imageDTO) {
      return {
        id: `node-${nodeId}-${field.name}`,
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO },
      };
    }
  }, [field.name, imageDTO, nodeId]);

  const droppableData = useMemo<TypesafeDroppableData | undefined>(
    () => ({
      id: `node-${nodeId}-${field.name}`,
      actionType: 'SET_NODES_IMAGE',
      context: { nodeId, fieldName: field.name },
    }),
    [field.name, nodeId]
  );

  const postUploadAction = useMemo<PostUploadAction>(
    () => ({
      type: 'SET_NODES_IMAGE',
      nodeId,
      fieldName: field.name,
    }),
    [nodeId, field.name]
  );

  useEffect(() => {
    if (isConnected && isError) {
      handleReset();
    }
  }, [handleReset, isConnected, isError]);

  return (
    <Flex className="nodrag" w="full" h="full" alignItems="center" justifyContent="center">
      <IAIDndImage
        imageDTO={imageDTO}
        droppableData={droppableData}
        draggableData={draggableData}
        postUploadAction={postUploadAction}
        useThumbailFallback
        uploadElement={<UploadElement />}
        dropLabel={<DropLabel />}
        minSize={8}
      >
        <IAIDndImageIcon
          onClick={handleReset}
          icon={imageDTO ? <PiArrowCounterClockwiseBold /> : undefined}
          tooltip="Reset Image"
        />
      </IAIDndImage>
    </Flex>
  );
};

export default memo(ImageFieldInputComponent);

const UploadElement = memo(() => {
  const { t } = useTranslation();
  return (
    <Text fontSize={16} fontWeight="semibold">
      {t('gallery.dropOrUpload')}
    </Text>
  );
});

UploadElement.displayName = 'UploadElement';

const DropLabel = memo(() => {
  const { t } = useTranslation();
  return (
    <Text fontSize={16} fontWeight="semibold">
      {t('gallery.drop')}
    </Text>
  );
});

DropLabel.displayName = 'DropLabel';
