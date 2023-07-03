import { useAppDispatch } from 'app/store/storeHooks';

import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import {
  ImageCollectionInputFieldTemplate,
  ImageCollectionInputFieldValue,
} from 'features/nodes/types/types';
import { memo, useCallback } from 'react';

import { FieldComponentProps } from './types';
import IAIDndImage from 'common/components/IAIDndImage';
import { ImageDTO } from 'services/api/types';
import { Flex } from '@chakra-ui/react';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { skipToken } from '@reduxjs/toolkit/dist/query';
import { uniq, uniqBy } from 'lodash-es';
import {
  NodesMultiImageDropData,
  isValidDrop,
  useDroppable,
} from 'app/components/ImageDnd/typesafeDnd';
import IAIDropOverlay from 'common/components/IAIDropOverlay';

const ImageCollectionInputFieldComponent = (
  props: FieldComponentProps<
    ImageCollectionInputFieldValue,
    ImageCollectionInputFieldTemplate
  >
) => {
  const { nodeId, field } = props;

  const dispatch = useAppDispatch();

  const handleDrop = useCallback(
    ({ image_name }: ImageDTO) => {
      dispatch(
        fieldValueChanged({
          nodeId,
          fieldName: field.name,
          value: uniqBy([...(field.value ?? []), { image_name }], 'image_name'),
        })
      );
    },
    [dispatch, field.name, field.value, nodeId]
  );

  const droppableData: NodesMultiImageDropData = {
    id: `node-${nodeId}-${field.name}`,
    actionType: 'SET_MULTI_NODES_IMAGE',
    context: { nodeId, fieldName: field.name },
  };

  const {
    isOver,
    setNodeRef: setDroppableRef,
    active,
    over,
  } = useDroppable({
    id: `node_${nodeId}`,
    data: droppableData,
  });

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
      ref={setDroppableRef}
      sx={{
        w: 'full',
        h: 'full',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
        minH: '10rem',
      }}
    >
      {field.value?.map(({ image_name }) => (
        <ImageSubField key={image_name} imageName={image_name} />
      ))}
      {isValidDrop(droppableData, active) && <IAIDropOverlay isOver={isOver} />}
    </Flex>
  );
};

export default memo(ImageCollectionInputFieldComponent);

type ImageSubFieldProps = { imageName: string };

const ImageSubField = (props: ImageSubFieldProps) => {
  const { currentData: image } = useGetImageDTOQuery(props.imageName);

  return (
    <IAIDndImage imageDTO={image} isDropDisabled={true} isDragDisabled={true} />
  );
};
