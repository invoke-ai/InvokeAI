import '@fontsource/inter/index.css';
import '@platform/i18n/client';
import { identityTransportAuthAdapter } from '@features/identity';
import { configureHttpAuth } from '@platform/transport/http';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import { App } from './App';

// The only production binding that must exist before React mounts; all other
// bindings are enumerated in `app/workbenchPorts.tsx`.
configureHttpAuth(identityTransportAuthAdapter);

const rootElement = document.getElementById('root');

if (!rootElement) {
  throw new Error('Unable to mount Invoke V7: root element was not found.');
}

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>
);
