import 'i18n';

import type { Middleware } from '@reduxjs/toolkit';
import type { StudioInitAction } from 'app/hooks/useStudioInitAction';
import { $didStudioInit } from 'app/hooks/useStudioInitAction';
import type { LoggingOverrides } from 'app/logging/logger';
import { $loggingOverrides, configureLogging } from 'app/logging/logger';
import {
  $resetClientState,
  buildDriver,
  buildResetClientState,
  type StorageDriverApi,
} from 'app/store/enhancers/reduxRemember/driver';
import { $accountSettingsLink } from 'app/store/nanostores/accountSettingsLink';
import { $authToken } from 'app/store/nanostores/authToken';
import { $baseUrl } from 'app/store/nanostores/baseUrl';
import { $customNavComponent } from 'app/store/nanostores/customNavComponent';
import type { CustomStarUi } from 'app/store/nanostores/customStarUI';
import { $customStarUI } from 'app/store/nanostores/customStarUI';
import { $isDebugging } from 'app/store/nanostores/isDebugging';
import { $logo } from 'app/store/nanostores/logo';
import { $onClickGoToModelManager } from 'app/store/nanostores/onClickGoToModelManager';
import { $openAPISchemaUrl } from 'app/store/nanostores/openAPISchemaUrl';
import { $projectId, $projectName, $projectUrl } from 'app/store/nanostores/projectId';
import { $queueId, DEFAULT_QUEUE_ID } from 'app/store/nanostores/queueId';
import { $store } from 'app/store/nanostores/store';
import { $toastMap } from 'app/store/nanostores/toastMap';
import { $whatsNew } from 'app/store/nanostores/whatsNew';
import { createStore } from 'app/store/store';
import type { PartialAppConfig } from 'app/types/invokeai';
import Loading from 'common/components/Loading/Loading';
import type { WorkflowSortOption, WorkflowTagCategory } from 'features/nodes/store/workflowLibrarySlice';
import {
  $workflowLibraryCategoriesOptions,
  $workflowLibrarySortOptions,
  $workflowLibraryTagCategoriesOptions,
  DEFAULT_WORKFLOW_LIBRARY_CATEGORIES,
  DEFAULT_WORKFLOW_LIBRARY_SORT_OPTIONS,
  DEFAULT_WORKFLOW_LIBRARY_TAG_CATEGORIES,
} from 'features/nodes/store/workflowLibrarySlice';
import type { WorkflowCategory } from 'features/nodes/types/workflow';
import type { ToastConfig } from 'features/toast/toast';
import type { PropsWithChildren, ReactNode } from 'react';
import React, { lazy, memo, useEffect, useLayoutEffect, useMemo } from 'react';
import { Provider } from 'react-redux';
import { addMiddleware, resetMiddlewares } from 'redux-dynamic-middlewares';
import { $socketOptions } from 'services/events/stores';
import type { ManagerOptions, SocketOptions } from 'socket.io-client';

const App = lazy(() => import('./App'));

interface Props extends PropsWithChildren {
  apiUrl?: string;
  openAPISchemaUrl?: string;
  token?: string;
  config?: PartialAppConfig;
  customNavComponent?: ReactNode;
  accountSettingsLink?: string;
  middleware?: Middleware[];
  projectId?: string;
  projectName?: string;
  projectUrl?: string;
  queueId?: string;
  studioInitAction?: StudioInitAction;
  customStarUi?: CustomStarUi;
  socketOptions?: Partial<ManagerOptions & SocketOptions>;
  isDebugging?: boolean;
  logo?: ReactNode;
  toastMap?: Record<string, ToastConfig>;
  whatsNew?: ReactNode[];
  workflowCategories?: WorkflowCategory[];
  workflowTagCategories?: WorkflowTagCategory[];
  workflowSortOptions?: WorkflowSortOption[];
  loggingOverrides?: LoggingOverrides;
  /**
   * If provided, overrides in-app navigation to the model manager
   */
  onClickGoToModelManager?: () => void;
  storageDriverApi?: StorageDriverApi;
}

const InvokeAIUI = ({
  apiUrl,
  openAPISchemaUrl,
  token,
  config,
  customNavComponent,
  accountSettingsLink,
  middleware,
  projectId,
  projectName,
  projectUrl,
  queueId,
  studioInitAction,
  customStarUi,
  socketOptions,
  isDebugging = false,
  logo,
  toastMap,
  workflowCategories,
  workflowTagCategories,
  workflowSortOptions,
  loggingOverrides,
  onClickGoToModelManager,
  whatsNew,
  storageDriverApi,
}: Props) => {
  useLayoutEffect(() => {
    /*
     * We need to configure logging before anything else happens - useLayoutEffect ensures we set this at the first
     * possible opportunity.
     *
     * Once redux initializes, we will check the user's settings and update the logging config accordingly. See
     * `useSyncLoggingConfig`.
     */
    $loggingOverrides.set(loggingOverrides);

    // Until we get the user's settings, we will use the overrides OR default values.
    configureLogging(
      loggingOverrides?.logIsEnabled ?? true,
      loggingOverrides?.logLevel ?? 'debug',
      loggingOverrides?.logNamespaces ?? '*'
    );
  }, [loggingOverrides]);

  useLayoutEffect(() => {
    if (studioInitAction) {
      $didStudioInit.set(false);
    }
  }, [studioInitAction]);

  useEffect(() => {
    // configure API client token
    if (token) {
      $authToken.set(token);
    }

    // configure API client base url
    if (apiUrl) {
      $baseUrl.set(apiUrl);
    }

    // configure API client project header
    if (projectId) {
      $projectId.set(projectId);
    }

    // configure API client project header
    if (queueId) {
      $queueId.set(queueId);
    }

    // reset dynamically added middlewares
    resetMiddlewares();

    // TODO: at this point, after resetting the middleware, we really ought to clean up the socket
    // stuff by calling `dispatch(socketReset())`. but we cannot dispatch from here as we are
    // outside the provider. it's not needed until there is the possibility that we will change
    // the `apiUrl`/`token` dynamically.

    // rebuild socket middleware with token and apiUrl
    if (middleware && middleware.length > 0) {
      addMiddleware(...middleware);
    }

    return () => {
      // Reset the API client token and base url on unmount
      $baseUrl.set(undefined);
      $authToken.set(undefined);
      $projectId.set(undefined);
      $queueId.set(DEFAULT_QUEUE_ID);
    };
  }, [apiUrl, token, middleware, projectId, queueId, projectName, projectUrl]);

  useEffect(() => {
    if (customStarUi) {
      $customStarUI.set(customStarUi);
    }

    return () => {
      $customStarUI.set(undefined);
    };
  }, [customStarUi]);

  useEffect(() => {
    if (customNavComponent) {
      $customNavComponent.set(customNavComponent);
    }

    return () => {
      $customNavComponent.set(undefined);
    };
  }, [customNavComponent]);

  useEffect(() => {
    if (accountSettingsLink) {
      $accountSettingsLink.set(accountSettingsLink);
    }

    return () => {
      $accountSettingsLink.set(undefined);
    };
  }, [accountSettingsLink]);

  useEffect(() => {
    if (openAPISchemaUrl) {
      $openAPISchemaUrl.set(openAPISchemaUrl);
    }

    return () => {
      $openAPISchemaUrl.set(undefined);
    };
  }, [openAPISchemaUrl]);

  useEffect(() => {
    $projectName.set(projectName);

    return () => {
      $projectName.set(undefined);
    };
  }, [projectName]);

  useEffect(() => {
    $projectUrl.set(projectUrl);

    return () => {
      $projectUrl.set(undefined);
    };
  }, [projectUrl]);

  useEffect(() => {
    if (logo) {
      $logo.set(logo);
    }

    return () => {
      $logo.set(undefined);
    };
  }, [logo]);

  useEffect(() => {
    if (toastMap) {
      $toastMap.set(toastMap);
    }

    return () => {
      $toastMap.set(undefined);
    };
  }, [toastMap]);

  useEffect(() => {
    if (whatsNew) {
      $whatsNew.set(whatsNew);
    }

    return () => {
      $whatsNew.set(undefined);
    };
  }, [whatsNew]);

  useEffect(() => {
    if (onClickGoToModelManager) {
      $onClickGoToModelManager.set(onClickGoToModelManager);
    }

    return () => {
      $onClickGoToModelManager.set(undefined);
    };
  }, [onClickGoToModelManager]);

  useEffect(() => {
    if (workflowCategories) {
      $workflowLibraryCategoriesOptions.set(workflowCategories);
    }

    return () => {
      $workflowLibraryCategoriesOptions.set(DEFAULT_WORKFLOW_LIBRARY_CATEGORIES);
    };
  }, [workflowCategories]);

  useEffect(() => {
    if (workflowTagCategories) {
      $workflowLibraryTagCategoriesOptions.set(workflowTagCategories);
    }

    return () => {
      $workflowLibraryTagCategoriesOptions.set(DEFAULT_WORKFLOW_LIBRARY_TAG_CATEGORIES);
    };
  }, [workflowTagCategories]);

  useEffect(() => {
    if (workflowSortOptions) {
      $workflowLibrarySortOptions.set(workflowSortOptions);
    }

    return () => {
      $workflowLibrarySortOptions.set(DEFAULT_WORKFLOW_LIBRARY_SORT_OPTIONS);
    };
  }, [workflowSortOptions]);

  useEffect(() => {
    if (socketOptions) {
      $socketOptions.set(socketOptions);
    }
    return () => {
      $socketOptions.set({});
    };
  }, [socketOptions]);

  useEffect(() => {
    if (isDebugging) {
      $isDebugging.set(isDebugging);
    }
    return () => {
      $isDebugging.set(false);
    };
  }, [isDebugging]);

  useEffect(() => {
    $resetClientState.set(buildResetClientState(storageDriverApi));

    return () => {
      $resetClientState.set(() => {});
    };
  }, [storageDriverApi]);

  const store = useMemo(() => {
    const driver = buildDriver(storageDriverApi);
    return createStore(driver);
  }, [storageDriverApi]);

  useEffect(() => {
    $store.set(store);
    if (import.meta.env.MODE === 'development') {
      window.$store = $store;
    }
    return () => {
      $store.set(undefined);
      if (import.meta.env.MODE === 'development') {
        window.$store = undefined;
      }
    };
  }, [store]);

  return (
    <React.StrictMode>
      <Provider store={store}>
        <React.Suspense fallback={<Loading />}>
          <App config={config} studioInitAction={studioInitAction} />
        </React.Suspense>
      </Provider>
    </React.StrictMode>
  );
};

export default memo(InvokeAIUI);
