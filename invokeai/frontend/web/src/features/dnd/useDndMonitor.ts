import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { monitorForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { logger } from 'app/logging/logger';
import { getStore } from 'app/store/nanostores/store';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { parseify } from 'common/util/serialize';
import { dndTargets, multipleImageDndSource, singleImageDndSource } from 'features/dnd/dnd';
import { useEffect } from 'react';

const log = logger('dnd');

export const useDndMonitor = () => {
  useAssertSingleton('useDropMonitor');

  useEffect(() => {
    return combine(
      monitorForElements({
        canMonitor: ({ source }) => {
          const sourceData = source.data;

          // Check for allowed sources
          if (!singleImageDndSource.typeGuard(sourceData) && !multipleImageDndSource.typeGuard(sourceData)) {
            return false;
          }

          return true;
        },
        onDrop: ({ source, location }) => {
          const target = location.current.dropTargets[0];
          if (!target) {
            log.warn('No dnd target');
            return;
          }

          const sourceData = source.data;
          const targetData = target.data;

          const { dispatch, getState } = getStore();

          for (const dndTarget of dndTargets) {
            if (!dndTarget.typeGuard(targetData)) {
              continue;
            }
            const arg = { sourceData, targetData, dispatch, getState };
            // TS cannot infer `arg.targetData` but we've just checked it.
            // TODO(psyche): Figure out how to satisfy TS.
            /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
            if (!dndTarget.isValid(arg as any)) {
              continue;
            }

            log.debug(parseify({ sourceData, targetData }), 'Handling dnd drop');

            // TS cannot infer `arg.targetData` but we've just checked it.
            // TODO(psyche): Figure out how to satisfy TS.
            /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
            dndTarget.handler(arg as any);
            return;
          }

          log.warn(parseify({ sourceData, targetData }), 'Invalid drop');
        },
      })
    );
  }, []);
};
