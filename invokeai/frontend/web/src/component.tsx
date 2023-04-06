import React, { lazy, PropsWithChildren, useEffect, useState } from 'react';
import { Provider } from 'react-redux';
import { PersistGate } from 'redux-persist/integration/react';
import { buildMiddleware, store } from './app/store';
import { persistor } from './persistor';
import { OpenAPI } from 'services/api';
import { InvokeTabName } from 'features/ui/store/tabMap';
import '@fontsource/inter/100.css';
import '@fontsource/inter/200.css';
import '@fontsource/inter/300.css';
import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@fontsource/inter/700.css';
import '@fontsource/inter/800.css';
import '@fontsource/inter/900.css';

import Loading from './Loading';

// Localization
import './i18n';
import { addMiddleware, resetMiddlewares } from 'redux-dynamic-middlewares';

const App = lazy(() => import('./app/App'));
const ThemeLocaleProvider = lazy(() => import('./app/ThemeLocaleProvider'));

interface Props extends PropsWithChildren {
  apiUrl?: string;
  disabledPanels?: string[];
  disabledTabs?: InvokeTabName[];
  token?: string;
}

export default function Component({
  apiUrl,
  disabledPanels = [],
  disabledTabs = [],
  token,
  children,
}: Props) {
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

    // rebuild socket middleware with token and apiUrl
    addMiddleware(buildMiddleware());
  }, [apiUrl, token]);

  return (
    <React.StrictMode>
      <Provider store={store}>
        <PersistGate loading={<Loading />} persistor={persistor}>
          <React.Suspense fallback={<Loading showText />}>
            <ThemeLocaleProvider>
              <App options={{ disabledPanels, disabledTabs }}>{children}</App>
            </ThemeLocaleProvider>
          </React.Suspense>
        </PersistGate>
      </Provider>
    </React.StrictMode>
  );
}
