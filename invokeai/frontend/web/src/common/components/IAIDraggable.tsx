import type { BoxProps } from '@invoke-ai/ui-library';
import { Box } from '@invoke-ai/ui-library';
import { useDraggableTypesafe } from 'features/dnd/hooks/typesafeHooks';
import type { TypesafeDraggableData } from 'features/dnd/types';
import { memo, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';

type IAIDraggableProps = BoxProps & {
  disabled?: boolean;
  data?: TypesafeDraggableData;
};

const IAIDraggable = (props: IAIDraggableProps) => {
  const { data, disabled, ...rest } = props;
  const dndId = useRef(uuidv4());

  const { attributes, listeners, setNodeRef } = useDraggableTypesafe({
    id: dndId.current,
    disabled,
    data,
  });

  return (
    <Box
      ref={setNodeRef}
      position="absolute"
      w="full"
      h="full"
      top={0}
      insetInlineStart={0}
      {...attributes}
      {...listeners}
      {...rest}
    />
  );
};

export default memo(IAIDraggable);
