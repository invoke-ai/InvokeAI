import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { dropTargetForElements, monitorForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box } from '@invoke-ai/ui-library';
import { getStore } from 'app/store/nanostores/store';
import { useAppDispatch } from 'app/store/storeHooks';
import type { AnyDndTarget } from 'features/dnd/dnd';
import { DndDropOverlay } from 'features/dnd/DndDropOverlay';
import type { DndTargetState } from 'features/dnd/types';
import { memo, useEffect, useRef, useState } from 'react';

const sx = {
  position: 'absolute',
  top: 0,
  right: 0,
  bottom: 0,
  left: 0,
  w: 'full',
  h: 'full',
  pointerEvents: 'auto',
  // We must disable pointer events when idle to prevent the overlay from blocking clicks
  '&[data-dnd-state="idle"]': {
    pointerEvents: 'none',
  },
} satisfies SystemStyleObject;

type Props<T extends AnyDndTarget> = {
  dndTarget: T;
  dndTargetData: ReturnType<T['getData']>;
  label: string;
  isDisabled?: boolean;
};

export const DndDropTarget = memo(<T extends AnyDndTarget>(props: Props<T>) => {
  const { dndTarget, dndTargetData, label, isDisabled } = props;
  const [dndState, setDndState] = useState<DndTargetState>('idle');
  const ref = useRef<HTMLDivElement>(null);
  const dispatch = useAppDispatch();

  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }
    if (isDisabled) {
      return;
    }

    const { dispatch, getState } = getStore();

    return combine(
      dropTargetForElements({
        element,
        canDrop: ({ source }) => {
          // TS cannot infer `dndTargetData` but we've just checked it.
          // TODO(psyche): Figure out how to satisfy TS.
          /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
          const arg = { sourceData: source.data, targetData: dndTargetData, dispatch, getState } as any;
          return dndTarget.isValid(arg);
        },
        onDragEnter: () => {
          setDndState('over');
        },
        onDragLeave: () => {
          setDndState('potential');
        },
        getData: () => dndTargetData,
      }),
      monitorForElements({
        canMonitor: ({ source }) => {
          // TS cannot infer `dndTargetData` but we've just checked it.
          // TODO(psyche): Figure out how to satisfy TS.
          /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
          const arg = { sourceData: source.data, targetData: dndTargetData, dispatch, getState } as any;
          return dndTarget.isValid(arg);
        },
        onDragStart: () => {
          setDndState('potential');
        },
        onDrop: () => {
          setDndState('idle');
        },
      })
    );
  }, [dispatch, isDisabled, dndTarget, dndTargetData]);

  return (
    <Box ref={ref} sx={sx} data-dnd-state={dndState}>
      <DndDropOverlay dndState={dndState} label={label} />
    </Box>
  );
});

DndDropTarget.displayName = 'DndDropTarget';
