import '@fontsource/inter/index.css';
import '@platform/i18n/client';
import { identityTransportAuthAdapter } from '@features/identity';
import { registerServiceWorker } from '@platform/pwa/registerServiceWorker';
import { configureHttpAuth } from '@platform/transport/http';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import { configureAppAccountLifecycle } from './accountLifecycle';
import { App } from './App';
import { configureAppLogging } from './loggingRuntime';

// Identity, logging and HTTP ownership must be configured before any route starts work.
configureAppAccountLifecycle();
configureAppLogging();
configureHttpAuth(identityTransportAuthAdapter);
registerServiceWorker();

const rootElement = document.getElementById('root');

if (!rootElement) {
  throw new Error('Unable to mount Invoke V7: root element was not found.');
}

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>
);
