import { Middleware } from '@reduxjs/toolkit';
import { $customStarUI, CustomStarUi } from 'app/store/nanostores/customStarUI';
import { $headerComponent } from 'app/store/nanostores/headerComponent';
import { createStore } from 'app/store/store';
import { PartialAppConfig } from 'app/types/invokeai';
import {
  $queueId,
  DEFAULT_QUEUE_ID,
} from 'features/queue/store/queueNanoStore';
import React, {
  PropsWithChildren,
  ReactNode,
  lazy,
  memo,
  useEffect,
  useMemo,
} from 'react';
import { Provider } from 'react-redux';
import { addMiddleware, resetMiddlewares } from 'redux-dynamic-middlewares';
import { $authToken, $baseUrl, $projectId } from 'services/api/client';
import { socketMiddleware } from 'services/events/middleware';
import Loading from '../../common/components/Loading/Loading';
import AppDndContext from '../../features/dnd/components/AppDndContext';
import '../../i18n';
import { $store } from '../store/nanostores/store';

const App = lazy(() => import('./App'));
const ThemeLocaleProvider = lazy(() => import('./ThemeLocaleProvider'));

interface Props extends PropsWithChildren {
  apiUrl?: string;
  token?: string;
  config?: PartialAppConfig;
  headerComponent?: ReactNode;
  middleware?: Middleware[];
  projectId?: string;
  queueId?: string;
  selectedImage?: {
    imageName: string;
    action: 'sendToImg2Img' | 'sendToCanvas' | 'useAllParameters';
  };
  customStarUi?: CustomStarUi;
}

const InvokeAIUI = ({
  apiUrl,
  token,
  config,
  headerComponent,
  middleware,
  projectId,
  queueId,
  selectedImage,
  customStarUi,
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
      addMiddleware(socketMiddleware(), ...middleware);
    } else {
      addMiddleware(socketMiddleware());
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
    if (headerComponent) {
      $headerComponent.set(headerComponent);
    }

    return () => {
      $headerComponent.set(undefined);
    };
  }, [headerComponent]);

  const store = useMemo(() => {
    return createStore(projectId);
  }, [projectId]);

  useEffect(() => {
    $store.set(store);
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
