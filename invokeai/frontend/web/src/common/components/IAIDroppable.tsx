import { Box } from '@chakra-ui/react';
import {
  TypesafeDroppableData,
  isValidDrop,
  useDroppable,
} from 'app/components/ImageDnd/typesafeDnd';
import { AnimatePresence } from 'framer-motion';
import { memo, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import IAIDropOverlay from './IAIDropOverlay';

type IAIDroppableProps = {
  dropLabel?: string;
  disabled?: boolean;
  data?: TypesafeDroppableData;
};

const IAIDroppable = (props: IAIDroppableProps) => {
  const { dropLabel, data, disabled } = props;
  const dndId = useRef(uuidv4());

  const { isOver, setNodeRef, active } = useDroppable({
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
