import { useGlobalModifiersInit } from '@invoke-ai/ui-library';
import type { StudioInitAction } from 'app/hooks/useStudioInitAction';
import { useStudioInitAction } from 'app/hooks/useStudioInitAction';
import { useSyncQueueStatus } from 'app/hooks/useSyncQueueStatus';
import { useLogger } from 'app/logging/useLogger';
import { useSyncLoggingConfig } from 'app/logging/useSyncLoggingConfig';
import { appStarted } from 'app/store/middleware/listenerMiddleware/listeners/appStarted';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { PartialAppConfig } from 'app/types/invokeai';
import { useFocusRegionWatcher } from 'common/hooks/focus';
import { useCloseChakraTooltipsOnDragFix } from 'common/hooks/useCloseChakraTooltipsOnDragFix';
import { useGlobalHotkeys } from 'common/hooks/useGlobalHotkeys';
import { size } from 'es-toolkit/compat';
import { useDynamicPromptsWatcher } from 'features/dynamicPrompts/hooks/useDynamicPromptsWatcher';
import { useStarterModelsToast } from 'features/modelManagerV2/hooks/useStarterModelsToast';
import { useWorkflowBuilderWatcher } from 'features/nodes/components/sidePanel/workflow/IsolatedWorkflowBuilderWatcher';
import { useReadinessWatcher } from 'features/queue/store/readiness';
import { configChanged } from 'features/system/store/configSlice';
import { selectLanguage } from 'features/system/store/systemSelectors';
import { usePanelRegistryInit } from 'features/ui/layouts/panel-registry/use-panel-registry-init';
import i18n from 'i18n';
import { memo, useEffect } from 'react';
import { useGetOpenAPISchemaQuery } from 'services/api/endpoints/appInfo';
import { useGetQueueCountsByDestinationQuery } from 'services/api/endpoints/queue';
import { useSocketIO } from 'services/events/useSocketIO';

const queueCountArg = { destination: 'canvas' };

/**
 * GlobalHookIsolator is a logical component that runs global hooks in an isolated component, so that they do not
 * cause needless re-renders of any other components.
 */
export const GlobalHookIsolator = memo(
  ({ config, studioInitAction }: { config: PartialAppConfig; studioInitAction?: StudioInitAction }) => {
    const language = useAppSelector(selectLanguage);
    const logger = useLogger('system');
    const dispatch = useAppDispatch();

    // singleton!
    useReadinessWatcher();
    useSocketIO();
    useGlobalModifiersInit();
    useGlobalHotkeys();
    useGetOpenAPISchemaQuery();
    useSyncLoggingConfig();
    useCloseChakraTooltipsOnDragFix();
    usePanelRegistryInit();

    // Persistent subscription to the queue counts query - canvas relies on this to know if there are pending
    // and/or in progress canvas sessions.
    useGetQueueCountsByDestinationQuery(queueCountArg);

    useEffect(() => {
      i18n.changeLanguage(language);
    }, [language]);

    useEffect(() => {
      if (size(config)) {
        logger.info({ config }, 'Received config');
        dispatch(configChanged(config));
      }
    }, [dispatch, config, logger]);

    useEffect(() => {
      dispatch(appStarted());
    }, [dispatch]);

    useStudioInitAction(studioInitAction);
    useStarterModelsToast();
    useSyncQueueStatus();
    useFocusRegionWatcher();
    useWorkflowBuilderWatcher();
    useDynamicPromptsWatcher();

    return null;
  }
);
GlobalHookIsolator.displayName = 'GlobalHookIsolator';
