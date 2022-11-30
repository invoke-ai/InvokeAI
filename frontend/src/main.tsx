import React from 'react';
import ReactDOM from 'react-dom/client';
import { ChakraProvider } from '@chakra-ui/react';
import { CacheProvider } from '@emotion/react';
import createCache from '@emotion/cache';
import { store } from './app/store';
import { Provider } from 'react-redux';
import { PersistGate } from 'redux-persist/integration/react';
import { persistStore } from 'redux-persist';

export const persistor = persistStore(store);

import Loading from './Loading';
import App from './app/App';

export const emotionCache = createCache({
  key: 'invokeai-style-cache',
  prepend: true,
});

// Custom Styling
import './styles/index.scss';

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <Provider store={store}>
      <PersistGate loading={<Loading />} persistor={persistor}>
        <CacheProvider value={emotionCache}>
          <ChakraProvider>
            <App />
          </ChakraProvider>
        </CacheProvider>
      </PersistGate>
    </Provider>
  </React.StrictMode>
);
