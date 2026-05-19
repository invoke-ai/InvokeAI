import 'i18n';

import { configureLogging } from 'app/logging/logger';
import { addStorageListeners } from 'app/store/enhancers/reduxRemember/driver';
import { $store } from 'app/store/nanostores/store';
import { createStore } from 'app/store/store';
import Loading from 'common/components/Loading/Loading';
import React, { lazy, useEffect, useState } from 'react';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';

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
  const [didRehydrate, setDidRehydrate] = useState(false);
  const [store] = useState(() =>
    createStore({ persist: true, persistDebounce: 300, onRehydrated: () => setDidRehydrate(true) })
  );

  useEffect(() => {
    $store.set(store);
    if (import.meta.env.MODE === 'development') {
      window.$store = $store;
    }
    const removeStorageListeners = addStorageListeners();
    return () => {
      removeStorageListeners();
      $store.set(undefined);
      if (import.meta.env.MODE === 'development') {
        window.$store = undefined;
      }
    };
  }, [store]);

  if (!didRehydrate) {
    return <Loading />;
  }

  return (
    <Provider store={store}>
      <BrowserRouter>
        <React.Suspense fallback={<Loading />}>
          <App />
        </React.Suspense>
      </BrowserRouter>
    </Provider>
  );
};

export default InvokeAIUI;
