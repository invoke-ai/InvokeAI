import { ChakraProvider } from '@chakra-ui/react';
import createCache from '@emotion/cache';
import { CacheProvider } from '@emotion/react';
import React from 'react';
import ReactDOM from 'react-dom/client';
import { Provider } from 'react-redux';
import { PersistGate } from 'redux-persist/integration/react';
import { store } from './app/store';
import { persistor } from './persistor';

import App from './app/App';
import Loading from './Loading';

export const emotionCache = createCache({
  key: 'invokeai-style-cache',
  prepend: true,
});

// Custom Styling
import './styles/index.scss';

// Localization
import './i18n';

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <Provider store={store}>
      <PersistGate loading={<Loading />} persistor={persistor}>
        <CacheProvider value={emotionCache}>
          <ChakraProvider>
            <React.Suspense fallback={<Loading />}>
              <App />
            </React.Suspense>
          </ChakraProvider>
        </CacheProvider>
      </PersistGate>
    </Provider>
  </React.StrictMode>
);
