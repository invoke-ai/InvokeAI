import {
  DragOverlay,
  MouseSensor,
  TouchSensor,
  pointerWithin,
  useSensor,
  useSensors,
} from '@dnd-kit/core';
import { logger } from 'app/logging/logger';
import { dndDropped } from 'app/store/middleware/listenerMiddleware/listeners/imageDropped';
import { useAppDispatch } from 'app/store/storeHooks';
import { parseify } from 'common/util/serialize';
import { AnimatePresence, motion } from 'framer-motion';
import { PropsWithChildren, memo, useCallback, useState } from 'react';
import { useScaledModifer } from '../hooks/useScaledCenteredModifer';
import { DragEndEvent, DragStartEvent, TypesafeDraggableData } from '../types';
import { DndContextTypesafe } from './DndContextTypesafe';
import DragPreview from './DragPreview';

const AppDndContext = (props: PropsWithChildren) => {
  const [activeDragData, setActiveDragData] =
    useState<TypesafeDraggableData | null>(null);
  const log = logger('images');

  const dispatch = useAppDispatch();

  const handleDragStart = useCallback(
    (event: DragStartEvent) => {
      log.trace(
        { dragData: parseify(event.active.data.current) },
        'Drag started'
      );
      const activeData = event.active.data.current;
      if (!activeData) {
        return;
      }
      setActiveDragData(activeData);
    },
    [log]
  );

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      log.trace(
        { dragData: parseify(event.active.data.current) },
        'Drag ended'
      );
      const overData = event.over?.data.current;
      if (!activeDragData || !overData) {
        return;
      }
      dispatch(dndDropped({ overData, activeData: activeDragData }));
      setActiveDragData(null);
    },
    [activeDragData, dispatch, log]
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

  const scaledModifier = useScaledModifer();

  return (
    <DndContextTypesafe
      onDragStart={handleDragStart}
      onDragEnd={handleDragEnd}
      sensors={sensors}
      collisionDetection={pointerWithin}
      autoScroll={false}
    >
      {props.children}
      <DragOverlay
        dropAnimation={null}
        modifiers={[scaledModifier]}
        style={{
          width: 'min-content',
          height: 'min-content',
          cursor: 'none',
          userSelect: 'none',
          // expand overlay to prevent cursor from going outside it and displaying
          padding: '10rem',
        }}
      >
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
    </DndContextTypesafe>
  );
};

export default memo(AppDndContext);
