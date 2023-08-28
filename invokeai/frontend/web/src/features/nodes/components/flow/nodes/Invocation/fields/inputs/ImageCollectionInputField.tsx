import {
  ImageCollectionInputFieldTemplate,
  ImageCollectionInputFieldValue,
  FieldComponentProps,
} from 'features/nodes/types/types';
import { memo } from 'react';

import { Flex } from '@chakra-ui/react';
import IAIDndImage from 'common/components/IAIDndImage';
import IAIDropOverlay from 'common/components/IAIDropOverlay';
import { useDroppableTypesafe } from 'features/dnd/hooks/typesafeHooks';
import { NodesMultiImageDropData } from 'features/dnd/types';
import { isValidDrop } from 'features/dnd/util/isValidDrop';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

const ImageCollectionInputFieldComponent = (
  props: FieldComponentProps<
    ImageCollectionInputFieldValue,
    ImageCollectionInputFieldTemplate
  >
) => {
  const { nodeId, field } = props;

  // const dispatch = useAppDispatch();

  // const handleDrop = useCallback(
  //   ({ image_name }: ImageDTO) => {
  //     dispatch(
  //       fieldValueChanged({
  //         nodeId,
  //         fieldName: field.name,
  //         value: uniqBy([...(field.value ?? []), { image_name }], 'image_name'),
  //       })
  //     );
  //   },
  //   [dispatch, field.name, field.value, nodeId]
  // );

  const droppableData: NodesMultiImageDropData = {
    id: `node-${nodeId}-${field.name}`,
    actionType: 'SET_MULTI_NODES_IMAGE',
    context: { nodeId: nodeId, fieldName: field.name },
  };

  const {
    isOver,
    setNodeRef: setDroppableRef,
    active,
  } = useDroppableTypesafe({
    id: `node_${nodeId}`,
    data: droppableData,
  });

  // const handleReset = useCallback(() => {
  //   dispatch(
  //     fieldValueChanged({
  //       nodeId,
  //       fieldName: field.name,
  //       value: undefined,
  //     })
  //   );
  // }, [dispatch, field.name, nodeId]);

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
