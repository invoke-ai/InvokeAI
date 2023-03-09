import React, { lazy } from 'react';
import { Provider } from 'react-redux';
import { PersistGate } from 'redux-persist/integration/react';
import { store } from './app/store';
import { persistor } from './persistor';
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

const App = lazy(() => import('./app/App'));
const ThemeLocaleProvider = lazy(() => import('./app/ThemeLocaleProvider'));

export default function Component() {
  return (
    <React.StrictMode>
      <Provider store={store}>
        <PersistGate loading={<Loading />} persistor={persistor}>
          <React.Suspense fallback={<Loading showText />}>
            <ThemeLocaleProvider>
              <App />
            </ThemeLocaleProvider>
          </React.Suspense>
        </PersistGate>
      </Provider>
    </React.StrictMode>
  );
}
