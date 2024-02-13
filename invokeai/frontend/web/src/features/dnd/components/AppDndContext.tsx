import { MouseSensor, TouchSensor, useSensor, useSensors } from '@dnd-kit/core';
import { logger } from 'app/logging/logger';
import { dndDropped } from 'app/store/middleware/listenerMiddleware/listeners/imageDropped';
import { useAppDispatch } from 'app/store/storeHooks';
import { parseify } from 'common/util/serialize';
import DndOverlay from 'features/dnd/components/DndOverlay';
import type { DragEndEvent, DragStartEvent, TypesafeDraggableData } from 'features/dnd/types';
import { customPointerWithin } from 'features/dnd/util/customPointerWithin';
import type { PropsWithChildren } from 'react';
import { memo, useCallback, useState } from 'react';

import { DndContextTypesafe } from './DndContextTypesafe';

const AppDndContext = (props: PropsWithChildren) => {
  const [activeDragData, setActiveDragData] = useState<TypesafeDraggableData | null>(null);
  const log = logger('images');

  const dispatch = useAppDispatch();

  const handleDragStart = useCallback(
    (event: DragStartEvent) => {
      console.log('handling drag start', event.active.data.current);
      log.trace({ dragData: parseify(event.active.data.current) }, 'Drag started');
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
      console.log('handling drag end', event.active.data.current);
      log.trace({ dragData: parseify(event.active.data.current) }, 'Drag ended');
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

  return (
    <DndContextTypesafe
      onDragStart={handleDragStart}
      onDragEnd={handleDragEnd}
      sensors={sensors}
      collisionDetection={customPointerWithin}
      autoScroll={false}
    >
      {props.children}
      <DndOverlay activeDragData={activeDragData} />
    </DndContextTypesafe>
  );
};

export default memo(AppDndContext);
