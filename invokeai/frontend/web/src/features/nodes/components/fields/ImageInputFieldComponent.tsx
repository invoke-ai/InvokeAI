import { useAppDispatch } from 'app/store/storeHooks';

import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import {
  ImageInputFieldTemplate,
  ImageInputFieldValue,
} from 'features/nodes/types/types';
import { memo, useCallback } from 'react';

import { FieldComponentProps } from './types';
import IAIDndImage from 'common/components/IAIDndImage';
import { ImageDTO } from 'services/api';
import { Flex } from '@chakra-ui/react';
import { useGetImageDTOQuery } from 'services/apiSlice';
import { skipToken } from '@reduxjs/toolkit/dist/query';

const ImageInputFieldComponent = (
  props: FieldComponentProps<ImageInputFieldValue, ImageInputFieldTemplate>
) => {
  const { nodeId, field } = props;

  const dispatch = useAppDispatch();

  const {
    data: image,
    isLoading,
    isError,
    isSuccess,
  } = useGetImageDTOQuery(field.value ?? skipToken);

  const handleDrop = useCallback(
    (droppedImage: ImageDTO) => {
      if (field.value === droppedImage.image_name) {
        return;
      }

      dispatch(
        fieldValueChanged({
          nodeId,
          fieldName: field.name,
          value: droppedImage.image_name,
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
        image={image}
        onDrop={handleDrop}
        onReset={handleReset}
        resetIconSize="sm"
        postUploadAction={{
          type: 'SET_NODES_IMAGE',
          nodeId,
          fieldName: field.name,
        }}
      />
    </Flex>
  );
};

export default memo(ImageInputFieldComponent);
