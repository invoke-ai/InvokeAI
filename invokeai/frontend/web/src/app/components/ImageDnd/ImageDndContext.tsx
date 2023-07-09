import {
  DragOverlay,
  MouseSensor,
  TouchSensor,
  pointerWithin,
  useSensor,
  useSensors,
} from '@dnd-kit/core';
import { snapCenterToCursor } from '@dnd-kit/modifiers';
import { imageDropped } from 'app/store/middleware/listenerMiddleware/listeners/imageDropped';
import { useAppDispatch } from 'app/store/storeHooks';
import { AnimatePresence, motion } from 'framer-motion';
import { PropsWithChildren, memo, useCallback, useState } from 'react';
import DragPreview from './DragPreview';
import {
  DndContext,
  DragEndEvent,
  DragStartEvent,
  TypesafeDraggableData,
} from './typesafeDnd';

type ImageDndContextProps = PropsWithChildren;

const ImageDndContext = (props: ImageDndContextProps) => {
  const [activeDragData, setActiveDragData] =
    useState<TypesafeDraggableData | null>(null);

  const dispatch = useAppDispatch();

  const handleDragStart = useCallback((event: DragStartEvent) => {
    const activeData = event.active.data.current;
    if (!activeData) {
      return;
    }
    setActiveDragData(activeData);
  }, []);

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      const activeData = event.active.data.current;
      const overData = event.over?.data.current;
      if (!activeData || !overData) {
        return;
      }
      dispatch(imageDropped({ overData, activeData }));
      setActiveDragData(null);
    },
    [dispatch]
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
      <DragOverlay dropAnimation={null} modifiers={[snapCenterToCursor]}>
        <AnimatePresence>
          {activeDragData && (
            <motion.div
              layout
              key="overlay-drag-image"
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
              <DragPreview dragData={activeDragData} />
            </motion.div>
          )}
        </AnimatePresence>
      </DragOverlay>
    </DndContext>
  );
};

export default memo(ImageDndContext);
