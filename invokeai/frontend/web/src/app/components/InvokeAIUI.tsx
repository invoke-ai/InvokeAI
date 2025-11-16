import 'i18n';

import { configureLogging } from 'app/logging/logger';
import { addStorageListeners } from 'app/store/enhancers/reduxRemember/driver';
import { $store } from 'app/store/nanostores/store';
import { createStore } from 'app/store/store';
import Loading from 'common/components/Loading/Loading';
import React, { lazy, memo, useEffect, useState } from 'react';
import { Provider } from 'react-redux';

/*
 * We need to configure logging before anything else happens - useLayoutEffect ensures we set this at the first
 * possible opportunity.
 *
 * Once redux initializes, we will check the user's settings and update the logging config accordingly. See
 * `useSyncLoggingConfig`.
 */
configureLogging(true, 'debug', '*');

const App = lazy(() => import('./App'));

const InvokeAIUI = () => {
  const [store, setStore] = useState<ReturnType<typeof createStore> | undefined>(undefined);
  const [didRehydrate, setDidRehydrate] = useState(false);

  useEffect(() => {
    const onRehydrated = () => {
      setDidRehydrate(true);
    };
    const store = createStore({ persist: true, persistDebounce: 300, onRehydrated });
    setStore(store);
    $store.set(store);
    if (import.meta.env.MODE === 'development') {
      window.$store = $store;
    }
    const removeStorageListeners = addStorageListeners();
    return () => {
      removeStorageListeners();
      setStore(undefined);
      $store.set(undefined);
      if (import.meta.env.MODE === 'development') {
        window.$store = undefined;
      }
    };
  }, []);

  if (!store || !didRehydrate) {
    return <Loading />;
  }

  return (
    <React.StrictMode>
      <Provider store={store}>
        <React.Suspense fallback={<Loading />}>
          <App />
        </React.Suspense>
      </Provider>
    </React.StrictMode>
  );
};

export default memo(InvokeAIUI);
