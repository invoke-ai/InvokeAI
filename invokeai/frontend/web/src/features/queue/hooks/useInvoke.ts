import { useStore } from '@nanostores/react';
import { logger } from 'app/logging/logger';
import { useAppSelector } from 'app/store/storeHooks';
import { withResultAsync } from 'common/util/result';
import { useIsWorkflowEditorLocked } from 'features/nodes/hooks/useIsWorkflowEditorLocked';
import { useEnqueueWorkflows } from 'features/queue/hooks/useEnqueueWorkflows';
import { $isReadyToEnqueue } from 'features/queue/store/readiness';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { VIEWER_PANEL_ID, WORKSPACE_PANEL_ID } from 'features/ui/layouts/shared';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { useCallback } from 'react';
import { serializeError } from 'serialize-error';
import { enqueueMutationFixedCacheKeyOptions, useEnqueueBatchMutation } from 'services/api/endpoints/queue';

import { useEnqueueCanvas } from './useEnqueueCanvas';
import { useEnqueueGenerate } from './useEnqueueGenerate';
import { useEnqueueUpscaling } from './useEnqueueUpscaling';

const log = logger('generation');

export const useInvoke = () => {
  const tabName = useAppSelector(selectActiveTab);
  const isReady = useStore($isReadyToEnqueue);
  const isLocked = useIsWorkflowEditorLocked();
  const enqueueWorkflows = useEnqueueWorkflows();
  const enqueueCanvas = useEnqueueCanvas();
  const enqueueGenerate = useEnqueueGenerate();
  const enqueueUpscaling = useEnqueueUpscaling();

  const [_, { isLoading }] = useEnqueueBatchMutation({
    ...enqueueMutationFixedCacheKeyOptions,
    selectFromResult: ({ isLoading }) => ({ isLoading }),
  });

  const enqueue = useCallback(
    async (prepend: boolean, isApiValidationRun: boolean) => {
      if (!isReady) {
        return;
      }

      const result = await withResultAsync(async () => {
        switch (tabName) {
          case 'workflows':
            return await enqueueWorkflows(prepend, isApiValidationRun);
          case 'canvas':
            return await enqueueCanvas(prepend);
          case 'generate':
            return await enqueueGenerate(prepend);
          case 'upscaling':
            return await enqueueUpscaling(prepend);
          default:
            throw new Error(`No enqueue handler for tab: ${tabName}`);
        }
      });

      if (result.isErr()) {
        log.error({ error: serializeError(result.error) }, 'Failed to enqueue batch');
      }
    },
    [enqueueCanvas, enqueueGenerate, enqueueUpscaling, enqueueWorkflows, isReady, tabName]
  );

  const enqueueBack = useCallback(() => {
    enqueue(false, false);
    if (tabName === 'generate' || tabName === 'upscaling') {
      navigationApi.focusPanel(tabName, VIEWER_PANEL_ID);
    } else if (tabName === 'workflows') {
      // Only switch to viewer if the workflow editor is not currently active
      const workspace = navigationApi.getPanel('workflows', WORKSPACE_PANEL_ID);
      if (!workspace?.api.isActive) {
        navigationApi.focusPanel(tabName, VIEWER_PANEL_ID);
      }
    } else if (tabName === 'canvas') {
      navigationApi.focusPanel(tabName, WORKSPACE_PANEL_ID);
    }
  }, [enqueue, tabName]);

  const enqueueFront = useCallback(() => {
    enqueue(true, false);
    if (tabName === 'generate' || tabName === 'upscaling') {
      navigationApi.focusPanel(tabName, VIEWER_PANEL_ID);
    } else if (tabName === 'workflows') {
      // Only switch to viewer if the workflow editor is not currently active
      const workspace = navigationApi.getPanel('workflows', WORKSPACE_PANEL_ID);
      if (!workspace?.api.isActive) {
        navigationApi.focusPanel(tabName, VIEWER_PANEL_ID);
      }
    } else if (tabName === 'canvas') {
      navigationApi.focusPanel(tabName, WORKSPACE_PANEL_ID);
    }
  }, [enqueue, tabName]);

  return { enqueueBack, enqueueFront, isLoading, isDisabled: !isReady || isLocked, enqueue };
};
