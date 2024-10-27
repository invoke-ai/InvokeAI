import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { dropTargetForElements, monitorForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { Box } from '@invoke-ai/ui-library';
import { dndDropped } from 'app/store/middleware/listenerMiddleware/listeners/dnd';
import { useAppDispatch } from 'app/store/storeHooks';
import { DndDropOverlay } from 'features/dnd2/DndDropOverlay';
import type { DndState, DndTargetData } from 'features/dnd2/types';
import { isDndSourceData, isValidDrop } from 'features/dnd2/types';
import { memo, useEffect, useRef, useState } from 'react';

type Props = {
  label?: string;
  disabled?: boolean;
  targetData: DndTargetData;
};

export const DndDropTarget = memo((props: Props) => {
  const { label, targetData, disabled } = props;
  const [dndState, setDndState] = useState<DndState>('idle');
  const ref = useRef<HTMLDivElement>(null);
  const dispatch = useAppDispatch();

  useEffect(() => {
    if (!ref.current) {
      return;
    }

    return combine(
      dropTargetForElements({
        element: ref.current,
        canDrop: (args) => {
          if (disabled) {
            return false;
          }
          const sourceData = args.source.data;
          if (!isDndSourceData(sourceData)) {
            return false;
          }
          return isValidDrop(sourceData, targetData);
        },
        onDragEnter: () => {
          setDndState('active');
        },
        onDragLeave: () => {
          setDndState('pending');
        },
        getData: () => targetData,
        onDrop: (args) => {
          const sourceData = args.source.data;
          if (!isDndSourceData(sourceData)) {
            return;
          }
          dispatch(dndDropped({ sourceData, targetData }));
        },
      }),
      monitorForElements({
        canMonitor: (args) => {
          if (disabled) {
            return false;
          }
          const sourceData = args.source.data;
          if (!isDndSourceData(sourceData)) {
            return false;
          }
          return isValidDrop(sourceData, targetData);
        },
        onDragStart: () => {
          setDndState('pending');
        },
        onDrop: () => {
          setDndState('idle');
        },
      })
    );
  }, [targetData, disabled, dispatch]);

  return (
    <Box
      ref={ref}
      position="absolute"
      top={0}
      right={0}
      bottom={0}
      left={0}
      w="full"
      h="full"
      pointerEvents={dndState === 'idle' ? 'none' : 'auto'}
    >
      <DndDropOverlay dndState={dndState} label={label} />
    </Box>
  );
});

DndDropTarget.displayName = 'DndDropTarget';
