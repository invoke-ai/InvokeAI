import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { monitorForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { monitorForExternal } from '@atlaskit/pragmatic-drag-and-drop/external/adapter';
import { containsFiles } from '@atlaskit/pragmatic-drag-and-drop/external/file';
import { preventUnhandled } from '@atlaskit/pragmatic-drag-and-drop/prevent-unhandled';
import { logger } from 'app/logging/logger';
import { getStore } from 'app/store/nanostores/store';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { parseify } from 'common/util/serialize';
import { multipleImageSourceApi, multipleImageActions, singleImageSourceApi, singleImageActions } from 'features/imageActions/actions';
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
          if (!singleImageSourceApi.typeGuard(sourceData) && !multipleImageSourceApi.typeGuard(sourceData)) {
            return false;
          }

          return true;
        },
        onDrop: ({ source, location }) => {
          const target = location.current.dropTargets[0];
          if (!target) {
            return;
          }

          const sourceData = source.data;
          const targetData = target.data;

          const { dispatch, getState } = getStore();

          // Check for allowed sources
          if (singleImageSourceApi.typeGuard(sourceData)) {
            for (const target of singleImageActions) {
              if (target.typeGuard(targetData)) {
                // TS cannot infer `targetData` but we've just checked it. This is safe.
                /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
                if (target.isValid(sourceData, targetData as any, dispatch, getState)) {
                  /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
                  target.handler(sourceData, targetData as any, dispatch, getState);
                  log.debug(parseify({ sourceData, targetData }), 'Dropped single image');
                  return;
                }
              }
            }
          }

          if (multipleImageSourceApi.typeGuard(sourceData)) {
            for (const target of multipleImageActions) {
              if (target.typeGuard(targetData)) {
                // TS cannot infer `targetData` but we've just checked it. This is safe.
                /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
                if (target.isValid(sourceData, targetData as any, dispatch, getState)) {
                  /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
                  target.handler(sourceData, targetData as any, dispatch, getState);
                  log.debug(parseify({ sourceData, targetData }), 'Dropped multiple images');
                  return;
                }
              }
            }
          }

          log.warn(parseify({ sourceData, targetData }), 'Invalid image drop');
        },
      }),
      monitorForExternal({
        canMonitor: (args) => {
          if (!containsFiles(args)) {
            return false;
          }
          return true;
        },
        onDragStart: () => {
          preventUnhandled.start();
        },
        onDrop: () => {
          preventUnhandled.stop();
        },
      })
    );
  }, []);
};
