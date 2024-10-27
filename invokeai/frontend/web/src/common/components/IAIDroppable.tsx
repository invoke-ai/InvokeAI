import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { dropTargetForElements, monitorForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { Box } from '@invoke-ai/ui-library';
import { dndDropped } from 'app/store/middleware/listenerMiddleware/listeners/imageDropped';
import { useAppDispatch } from 'app/store/storeHooks';
import { useDroppableTypesafe } from 'features/dnd/hooks/typesafeHooks';
import type { TypesafeDroppableData } from 'features/dnd/types';
import { singleImageDndSource } from 'features/dnd2/types';
import { memo, useEffect, useRef, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';

import IAIDropOverlay from './IAIDropOverlay';

type IAIDroppableProps = {
  dropLabel?: string;
  disabled?: boolean;
  data?: TypesafeDroppableData;
};

const IAIDroppable = (props: IAIDroppableProps) => {
  const { dropLabel, data, disabled } = props;
  const [dndState, setDndState] = useState<'idle' | 'pending' | 'active'>('idle');
  const dndId = useRef(uuidv4());
  const ref = useRef<HTMLDivElement>(null);
  const dispatch = useAppDispatch();

  const { isOver, setNodeRef, active } = useDroppableTypesafe({
    id: dndId.current,
    disabled,
    data,
  });

  useEffect(() => {
    if (!ref.current) {
      console.log('no ref');
      return;
    }

    if (!data) {
      console.log('no data');
      return;
    }

    console.log('setting up droppable');

    return combine(
      dropTargetForElements({
        element: ref.current,
        canDrop: (args) => singleImageDndSource.typeGuard(args.source.data),
        onDragEnter: (args) => {
          console.log('onDragEnter', args);
          setDndState('active');
        },
        onDragLeave: (args) => {
          console.log('onDragEnter', args);
          setDndState('pending');
        },
        getData: (args) => data,
        onDrop: (args) => {
          if (!singleImageDndSource.typeGuard(args.source.data)) {
            return;
          }

          if (args.source.data.imageDTOs.length === 0) {
            return;
          }

          if (args.source.data.imageDTOs.length > 1) {
            dispatch(
              dndDropped({
                overData: data,
                activeData: { payloadType: 'IMAGE_DTO', id: 'asdf', payload: { imageDTO } },
              })
            );
            return;
          }

          const imageDTO = args.source.data.imageDTOs.at(0);

          if (!imageDTO) {
            return;
          }

          dispatch(
            dndDropped({
              overData: data,
              activeData: { payloadType: 'IMAGE_DTO', id: 'asdf', payload: { imageDTO } },
            })
          );
        },
      }),
      monitorForElements({
        canMonitor: (args) => singleImageDndSource.typeGuard(args.source.data),
        onDragStart: (args) => {
          console.log('onDragStart', args);
          if (!singleImageDndSource.typeGuard(args.source.data)) {
            return;
          }
          setDndState('pending');
        },
        onDrop: (args) => {
          console.log('onDrop', args);
          setDndState('idle');
        },
      })
    );
  }, [data, dispatch]);

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
      {dndState !== 'idle' && <IAIDropOverlay isOver={dndState === 'active'} label={dropLabel} />}
    </Box>
  );
};

export default memo(IAIDroppable);
