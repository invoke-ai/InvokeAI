import { Box } from '@invoke-ai/ui-library';
import { useDroppableTypesafe } from 'features/dnd/hooks/typesafeHooks';
import type { TypesafeDroppableData } from 'features/dnd/types';
import { isValidDrop } from 'features/dnd/util/isValidDrop';
import { AnimatePresence } from 'framer-motion';
import type { ReactNode } from 'react';
import { memo, useRef } from 'react';
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
      pointerEvents={active ? 'auto' : 'none'}
    >
      <AnimatePresence>
        {isValidDrop(data, active) && <IAIDropOverlay isOver={isOver} label={dropLabel} />}
      </AnimatePresence>
    </Box>
  );
};

export default memo(IAIDroppable);
