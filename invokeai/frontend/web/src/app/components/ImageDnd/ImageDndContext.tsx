import {
  DndContext,
  DragEndEvent,
  DragOverlay,
  DragStartEvent,
  KeyboardSensor,
  MouseSensor,
  TouchSensor,
  pointerWithin,
  useSensor,
  useSensors,
} from '@dnd-kit/core';
import { PropsWithChildren, memo, useCallback, useState } from 'react';
import OverlayDragImage from './OverlayDragImage';
import { ImageDTO } from 'services/api';
import { isImageDTO } from 'services/types/guards';
import { snapCenterToCursor } from '@dnd-kit/modifiers';

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
    activationConstraint: { delay: 250, tolerance: 5 },
  });

  const touchSensor = useSensor(TouchSensor, {
    activationConstraint: { delay: 250, tolerance: 5 },
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
        {draggedImage && <OverlayDragImage image={draggedImage} />}
      </DragOverlay>
    </DndContext>
  );
};

export default memo(ImageDndContext);
