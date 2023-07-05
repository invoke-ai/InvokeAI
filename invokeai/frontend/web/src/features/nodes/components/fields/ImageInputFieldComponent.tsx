import { useAppDispatch } from 'app/store/storeHooks';

import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import {
  ImageInputFieldTemplate,
  ImageInputFieldValue,
} from 'features/nodes/types/types';
import { memo, useCallback, useMemo } from 'react';

import { Flex } from '@chakra-ui/react';
import { skipToken } from '@reduxjs/toolkit/dist/query';
import {
  TypesafeDraggableData,
  TypesafeDroppableData,
} from 'app/components/ImageDnd/typesafeDnd';
import IAIDndImage from 'common/components/IAIDndImage';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { PostUploadAction } from 'services/api/thunks/image';
import { ImageDTO } from 'services/api/types';
import { FieldComponentProps } from './types';

const ImageInputFieldComponent = (
  props: FieldComponentProps<ImageInputFieldValue, ImageInputFieldTemplate>
) => {
  const { nodeId, field } = props;

  const dispatch = useAppDispatch();

  const {
    currentData: imageDTO,
    isLoading,
    isError,
    isSuccess,
  } = useGetImageDTOQuery(field.value?.image_name ?? skipToken);

  const handleDrop = useCallback(
    ({ image_name }: ImageDTO) => {
      if (field.value?.image_name === image_name) {
        return;
      }

      dispatch(
        fieldValueChanged({
          nodeId,
          fieldName: field.name,
          value: { image_name },
        })
      );
    },
    [dispatch, field.name, field.value, nodeId]
  );

  const handleReset = useCallback(() => {
    dispatch(
      fieldValueChanged({
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

  const droppableData = useMemo<TypesafeDroppableData | undefined>(() => {
    if (imageDTO) {
      return {
        id: `node-${nodeId}-${field.name}`,
        actionType: 'SET_NODES_IMAGE',
        context: { nodeId, fieldName: field.name },
      };
    }
  }, [field.name, imageDTO, nodeId]);

  const postUploadAction = useMemo<PostUploadAction>(
    () => ({
      type: 'SET_NODES_IMAGE',
      nodeId,
      fieldName: field.name,
    }),
    [nodeId, field.name]
  );

  return (
    <Flex
      sx={{
        w: 'full',
        h: 'full',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <IAIDndImage
        imageDTO={imageDTO}
        droppableData={droppableData}
        draggableData={draggableData}
        onClickReset={handleReset}
        postUploadAction={postUploadAction}
      />
    </Flex>
  );
};

export default memo(ImageInputFieldComponent);
