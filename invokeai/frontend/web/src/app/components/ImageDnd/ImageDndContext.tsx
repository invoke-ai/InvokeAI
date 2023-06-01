import {
  DndContext,
  DragEndEvent,
  DragOverlay,
  DragStartEvent,
} from '@dnd-kit/core';
import { PropsWithChildren, memo, useCallback, useState } from 'react';
import OverlayDragImage from './OverlayDragImage';
import { ImageDTO } from 'services/api';
import { isImageDTO } from 'services/types/guards';

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

  return (
    <DndContext onDragStart={handleDragStart} onDragEnd={handleDragEnd}>
      {props.children}
      <DragOverlay dropAnimation={null}>
        {draggedImage && <OverlayDragImage image={draggedImage} />}
      </DragOverlay>
    </DndContext>
  );
};

export default memo(ImageDndContext);
