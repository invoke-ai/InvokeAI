import 'i18n';

import type { Middleware } from '@reduxjs/toolkit';
import { $socketOptions } from 'app/hooks/useSocketIO';
import { $authToken } from 'app/store/nanostores/authToken';
import { $baseUrl } from 'app/store/nanostores/baseUrl';
import { $customNavComponent } from 'app/store/nanostores/customNavComponent';
import type { CustomStarUi } from 'app/store/nanostores/customStarUI';
import { $customStarUI } from 'app/store/nanostores/customStarUI';
import { $galleryHeader } from 'app/store/nanostores/galleryHeader';
import { $isDebugging } from 'app/store/nanostores/isDebugging';
import { $logo } from 'app/store/nanostores/logo';
import { $openAPISchemaUrl } from 'app/store/nanostores/openAPISchemaUrl';
import { $projectId } from 'app/store/nanostores/projectId';
import { $queueId, DEFAULT_QUEUE_ID } from 'app/store/nanostores/queueId';
import { $store } from 'app/store/nanostores/store';
import { $workflowCategories } from 'app/store/nanostores/workflowCategories';
import { createStore } from 'app/store/store';
import type { PartialAppConfig } from 'app/types/invokeai';
import Loading from 'common/components/Loading/Loading';
import AppDndContext from 'features/dnd/components/AppDndContext';
import type { WorkflowCategory } from 'features/nodes/types/workflow';
import type { PropsWithChildren, ReactNode } from 'react';
import React, { lazy, memo, useEffect, useMemo } from 'react';
import { Provider } from 'react-redux';
import { addMiddleware, resetMiddlewares } from 'redux-dynamic-middlewares';
import type { ManagerOptions, SocketOptions } from 'socket.io-client';

const App = lazy(() => import('./App'));
const ThemeLocaleProvider = lazy(() => import('./ThemeLocaleProvider'));

interface Props extends PropsWithChildren {
  apiUrl?: string;
  openAPISchemaUrl?: string;
  token?: string;
  config?: PartialAppConfig;
  customNavComponent?: ReactNode;
  middleware?: Middleware[];
  projectId?: string;
  galleryHeader?: ReactNode;
  queueId?: string;
  selectedImage?: {
    imageName: string;
    action: 'sendToImg2Img' | 'sendToCanvas' | 'useAllParameters';
  };
  customStarUi?: CustomStarUi;
  socketOptions?: Partial<ManagerOptions & SocketOptions>;
  isDebugging?: boolean;
  logo?: ReactNode;
  workflowCategories?: WorkflowCategory[];
}

const InvokeAIUI = ({
  apiUrl,
  openAPISchemaUrl,
  token,
  config,
  customNavComponent,
  middleware,
  projectId,
  galleryHeader,
  queueId,
  selectedImage,
  customStarUi,
  socketOptions,
  isDebugging = false,
  logo,
  workflowCategories,
}: Props) => {
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
  }, [apiUrl, token, middleware, projectId, queueId]);

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
    if (openAPISchemaUrl) {
      $openAPISchemaUrl.set(openAPISchemaUrl);
    }

    return () => {
      $openAPISchemaUrl.set(undefined);
    };
  }, [openAPISchemaUrl]);

  useEffect(() => {
    if (galleryHeader) {
      $galleryHeader.set(galleryHeader);
    }

    return () => {
      $galleryHeader.set(undefined);
    };
  }, [galleryHeader]);

  useEffect(() => {
    if (logo) {
      $logo.set(logo);
    }

    return () => {
      $logo.set(undefined);
    };
  }, [logo]);

  useEffect(() => {
    if (workflowCategories) {
      $workflowCategories.set(workflowCategories);
    }

    return () => {
      $workflowCategories.set([]);
    };
  }, [workflowCategories]);

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

  const store = useMemo(() => {
    return createStore(projectId);
  }, [projectId]);

  useEffect(() => {
    $store.set(store);
    if (import.meta.env.MODE === 'development') {
      window.$store = $store;
    }
    () => {
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
          <ThemeLocaleProvider>
            <AppDndContext>
              <App config={config} selectedImage={selectedImage} />
            </AppDndContext>
          </ThemeLocaleProvider>
        </React.Suspense>
      </Provider>
    </React.StrictMode>
  );
};

export default memo(InvokeAIUI);
