import { useGlobalModifiersInit } from '@invoke-ai/ui-library';
import { setupListeners } from '@reduxjs/toolkit/query';
import { useSyncLangDirection } from 'app/hooks/useSyncLangDirection';
import { useSyncQueueStatus } from 'app/hooks/useSyncQueueStatus';
import { useSyncLoggingConfig } from 'app/logging/useSyncLoggingConfig';
import { appStarted } from 'app/store/middleware/listenerMiddleware/listeners/appStarted';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useFocusRegionWatcher } from 'common/hooks/focus';
import { useCloseChakraTooltipsOnDragFix } from 'common/hooks/useCloseChakraTooltipsOnDragFix';
import { useGlobalHotkeys } from 'common/hooks/useGlobalHotkeys';
import { useDndMonitor } from 'features/dnd/useDndMonitor';
import { useDynamicPromptsWatcher } from 'features/dynamicPrompts/hooks/useDynamicPromptsWatcher';
import { useStarterModelsToast } from 'features/modelManagerV2/hooks/useStarterModelsToast';
import { useWorkflowBuilderWatcher } from 'features/nodes/components/sidePanel/workflow/IsolatedWorkflowBuilderWatcher';
import { useSyncExecutionState } from 'features/nodes/hooks/useNodeExecutionState';
import { useSyncNodeErrors } from 'features/nodes/store/util/fieldValidators';
import { useReadinessWatcher } from 'features/queue/store/readiness';
import { selectLanguage } from 'features/system/store/systemSelectors';
import { useNavigationApi } from 'features/ui/layouts/use-navigation-api';
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
export const GlobalHookIsolator = memo(() => {
  const language = useAppSelector(selectLanguage);
  const dispatch = useAppDispatch();

  // singleton!
  useReadinessWatcher();
  useSocketIO();
  useGlobalModifiersInit();
  useGlobalHotkeys();
  useGetOpenAPISchemaQuery();
  useSyncLoggingConfig();
  useCloseChakraTooltipsOnDragFix();
  useNavigationApi();
  useDndMonitor();
  useSyncNodeErrors();
  useSyncLangDirection();

  // Persistent subscription to the queue counts query - canvas relies on this to know if there are pending
  // and/or in progress canvas sessions.
  useGetQueueCountsByDestinationQuery(queueCountArg);
  useSyncExecutionState();

  useEffect(() => {
    i18n.changeLanguage(language);
  }, [language]);

  useEffect(() => {
    dispatch(appStarted());
  }, [dispatch]);

  useEffect(() => {
    return setupListeners(dispatch);
  }, [dispatch]);

  useStarterModelsToast();
  useSyncQueueStatus();
  useFocusRegionWatcher();
  useWorkflowBuilderWatcher();
  useDynamicPromptsWatcher();

  return null;
});
GlobalHookIsolator.displayName = 'GlobalHookIsolator';
