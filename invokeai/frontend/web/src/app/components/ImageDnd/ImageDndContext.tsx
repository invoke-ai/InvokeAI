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
import { PropsWithChildren, memo, useCallback, useState } from 'react';
import OverlayDragImage from './OverlayDragImage';
import { ImageDTO } from 'services/api/types';
import { isImageDTO } from 'services/api/guards';
import { snapCenterToCursor } from '@dnd-kit/modifiers';
import { AnimatePresence, motion } from 'framer-motion';

type ImageDndContextProps = PropsWithChildren;

const ImageDndContext = (props: ImageDndContextProps) => {
  const [draggedImage, setDraggedImage] = useState<ImageDTO | null>(null);

  const handleDragStart = useCallback((event: DragStartEvent) => {
    const dragData = event.active.data.current;
    if (dragData && 'image' in dragData && isImageDTO(dragData.image)) {
      setDraggedImage(dragData.image);
    }
  }, []);

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      const handleDrop = event.over?.data.current?.handleDrop;
      if (handleDrop && typeof handleDrop === 'function' && draggedImage) {
        handleDrop(draggedImage);
      }
      setDraggedImage(null);
    },
    [draggedImage]
  );

  const mouseSensor = useSensor(MouseSensor, {
    activationConstraint: { delay: 150, tolerance: 5 },
  });

  const touchSensor = useSensor(TouchSensor, {
    activationConstraint: { delay: 150, tolerance: 5 },
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
          {draggedImage && (
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
              <OverlayDragImage image={draggedImage} />
            </motion.div>
          )}
        </AnimatePresence>
      </DragOverlay>
    </DndContext>
  );
};

export default memo(ImageDndContext);
