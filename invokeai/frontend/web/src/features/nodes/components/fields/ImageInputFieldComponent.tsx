import { useAppDispatch } from 'app/store/storeHooks';

import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import {
  ImageInputFieldTemplate,
  ImageInputFieldValue,
} from 'features/nodes/types/types';
import { memo, useCallback } from 'react';

import { FieldComponentProps } from './types';
import IAIDndImage from 'features/controlNet/components/parameters/IAISelectableImage';
import { ImageDTO } from 'services/api';
import { Flex } from '@chakra-ui/react';

const ImageInputFieldComponent = (
  props: FieldComponentProps<ImageInputFieldValue, ImageInputFieldTemplate>
) => {
  const { nodeId, field } = props;

  const dispatch = useAppDispatch();

  const handleChange = useCallback(
    (image: ImageDTO) => {
      dispatch(
        fieldValueChanged({
          nodeId,
          fieldName: field.name,
          value: image,
        })
      );
    },
    [dispatch, field.name, nodeId]
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
        image={field.value}
        onDrop={handleChange}
        onReset={handleReset}
        resetIconSize="sm"
      />
    </Flex>
  );
};

export default memo(ImageInputFieldComponent);
