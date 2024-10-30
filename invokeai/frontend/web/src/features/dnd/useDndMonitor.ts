import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { monitorForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { monitorForExternal } from '@atlaskit/pragmatic-drag-and-drop/external/adapter';
import { containsFiles } from '@atlaskit/pragmatic-drag-and-drop/external/file';
import { preventUnhandled } from '@atlaskit/pragmatic-drag-and-drop/prevent-unhandled';
import { logger } from 'app/logging/logger';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { Dnd } from 'features/dnd/dnd';
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
          if (!Dnd.Source.singleImage.typeGuard(sourceData) && !Dnd.Source.multipleImage.typeGuard(sourceData)) {
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

          // Check for allowed sources
          if (!Dnd.Source.singleImage.typeGuard(sourceData) && !Dnd.Source.multipleImage.typeGuard(sourceData)) {
            return;
          }

          // Check for allowed targets
          if (!Dnd.Util.isDndTargetData(targetData)) {
            return;
          }

          log.debug({ sourceData, targetData }, 'Dropped image');

          Dnd.Util.handleDrop(sourceData, targetData);
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
