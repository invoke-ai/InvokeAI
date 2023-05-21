import React, {
  lazy,
  memo,
  PropsWithChildren,
  ReactNode,
  useEffect,
} from 'react';
import { Provider } from 'react-redux';
import { store } from 'app/store/store';
import { OpenAPI } from 'services/api';

import Loading from '../../common/components/Loading/Loading';
import { addMiddleware, resetMiddlewares } from 'redux-dynamic-middlewares';
import { PartialAppConfig } from 'app/types/invokeai';

import '../../i18n';
import { socketMiddleware } from 'services/events/middleware';
import { Middleware } from '@reduxjs/toolkit';

const App = lazy(() => import('./App'));
const ThemeLocaleProvider = lazy(() => import('./ThemeLocaleProvider'));

interface Props extends PropsWithChildren {
  apiUrl?: string;
  token?: string;
  config?: PartialAppConfig;
  headerComponent?: ReactNode;
  setIsReady?: (isReady: boolean) => void;
  middleware?: Middleware[];
}

const InvokeAIUI = ({
  apiUrl,
  token,
  config,
  headerComponent,
  setIsReady,
  middleware,
}: Props) => {
  useEffect(() => {
    // configure API client token
    if (token) {
      OpenAPI.TOKEN = token;
    }

    // configure API client base url
    if (apiUrl) {
      OpenAPI.BASE = apiUrl;
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
  }, [apiUrl, token, middleware]);

  return (
    <React.StrictMode>
      <Provider store={store}>
        <React.Suspense fallback={<Loading />}>
          <ThemeLocaleProvider>
            <App
              config={config}
              headerComponent={headerComponent}
              setIsReady={setIsReady}
            />
          </ThemeLocaleProvider>
        </React.Suspense>
      </Provider>
    </React.StrictMode>
  );
};

export default memo(InvokeAIUI);
