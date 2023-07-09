import { Box } from '@chakra-ui/react';
import {
  TypesafeDraggableData,
  useDraggable,
} from 'app/components/ImageDnd/typesafeDnd';
import { MouseEvent, memo, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';

type IAIDraggableProps = {
  disabled?: boolean;
  data?: TypesafeDraggableData;
  onClick?: (event: MouseEvent<HTMLDivElement>) => void;
};

const IAIDraggable = (props: IAIDraggableProps) => {
  const { data, disabled, onClick } = props;
  const dndId = useRef(uuidv4());

  const { attributes, listeners, setNodeRef } = useDraggable({
    id: dndId.current,
    disabled,
    data,
  });

  return (
    <Box
      onClick={onClick}
      ref={setNodeRef}
      position="absolute"
      w="full"
      h="full"
      {...attributes}
      {...listeners}
    />
  );
};

export default memo(IAIDraggable);
