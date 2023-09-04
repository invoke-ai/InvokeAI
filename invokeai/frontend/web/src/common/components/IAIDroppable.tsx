import { Box } from '@chakra-ui/react';
import { useDroppableTypesafe } from 'features/dnd/hooks/typesafeHooks';
import { TypesafeDroppableData } from 'features/dnd/types';
import { isValidDrop } from 'features/dnd/util/isValidDrop';
import { AnimatePresence } from 'framer-motion';
import { ReactNode, memo, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import IAIDropOverlay from './IAIDropOverlay';

type IAIDroppableProps = {
  dropLabel?: ReactNode;
  disabled?: boolean;
  data?: TypesafeDroppableData;
};

const IAIDroppable = (props: IAIDroppableProps) => {
  const { dropLabel, data, disabled } = props;
  const dndId = useRef(uuidv4());

  const { isOver, setNodeRef, active } = useDroppableTypesafe({
    id: dndId.current,
    disabled,
    data,
  });

  return (
    <Box
      ref={setNodeRef}
      position="absolute"
      top={0}
      insetInlineStart={0}
      w="full"
      h="full"
      pointerEvents="none"
    >
      <AnimatePresence>
        {isValidDrop(data, active) && (
          <IAIDropOverlay isOver={isOver} label={dropLabel} />
        )}
      </AnimatePresence>
    </Box>
  );
};

export default memo(IAIDroppable);
