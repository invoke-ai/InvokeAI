import { Box, Text } from '@chakra-ui/react';
import {
  DndContext,
  DragEndEvent,
  DragOverlay,
  DragStartEvent,
  MouseSensor,
  TouchSensor,
  pointerWithin,
  useSensor,
  useSensors,
} from '@dnd-kit/core';
import { logger } from 'app/logging/logger';
import { useAppDispatch } from 'app/store/storeHooks';
import { AnimatePresence, motion } from 'framer-motion';
import { PropsWithChildren, memo, useCallback, useState } from 'react';
import { z } from 'zod';
import { fieldIsExposedChanged } from '../store/nodesSlice';
import { snapCenterToCursor } from '@dnd-kit/modifiers';

const zActiveData = z.object({
  nodeId: z.string(),
  fieldName: z.string(),
});

type ActiveData = z.infer<typeof zActiveData>;

const NodeDndContext = (props: PropsWithChildren) => {
  const [activeDragData, setActiveDragData] = useState<ActiveData | null>(null);
  const log = logger('nodes');

  const dispatch = useAppDispatch();

  const handleDragStart = useCallback(
    (event: DragStartEvent) => {
      log.trace({ dragData: event.active.data.current }, 'Drag started');
      const activeData = event.active.data.current;
      const result = zActiveData.safeParse(activeData);
      if (!result.success) {
        return;
      }
      setActiveDragData(result.data);
    },
    [log]
  );

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      log.trace({ dragData: event.active.data.current }, 'Drag ended');
      const activeData = event.active.data.current;
      const result = zActiveData.safeParse(activeData);
      if (!result.success) {
        return;
      }
      dispatch(fieldIsExposedChanged({ ...result.data, isExposed: true }));
      setActiveDragData(null);
    },
    [dispatch, log]
  );

  const mouseSensor = useSensor(MouseSensor, {
    activationConstraint: { distance: 10 },
  });

  const touchSensor = useSensor(TouchSensor, {
    activationConstraint: { distance: 10 },
  });

  // TODO: Use KeyboardSensor - needs composition of multiple collisionDetection algos
  // Alternatively, fix `rectIntersection` collection detection to work with the drag overlay
  // (currently the drag element collision rect is not correctly calculated)
  // const keyboardSensor = useSensor(KeyboardSensor);

  const sensors = useSensors(mouseSensor, touchSensor);

  return (
    <DndContext
      onDragStart={handleDragStart}
      onDragEnd={handleDragEnd}
      sensors={sensors}
      collisionDetection={pointerWithin}
    >
      {props.children}
      {/* <DragOverlay dropAnimation={null}> */}
      <DragOverlay dropAnimation={null}>
        <AnimatePresence>
          {activeDragData && (
            <motion.div
              layout
              key="overlay-node-field"
              initial={{
                opacity: 0,
                scale: 0.7,
              }}
              animate={{
                opacity: 1,
                scale: 1,
                transition: { duration: 0.1 },
              }}
            >
              <Box sx={{ position: 'relative' }}>
                <Box
                  sx={{
                    position: 'absolute',
                    userSelect: 'none',
                    cursor: 'grabbing',
                    p: 2,
                    px: 3,
                    bg: 'base.300',
                    borderRadius: 'base',
                    boxShadow: 'dark-lg',
                  }}
                >
                  <Text>{activeDragData.fieldName}</Text>
                </Box>
              </Box>
            </motion.div>
          )}
        </AnimatePresence>
      </DragOverlay>
    </DndContext>
  );
};

export default memo(NodeDndContext);
