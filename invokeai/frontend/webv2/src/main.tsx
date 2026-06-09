import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import { App } from './App';

const rootElement = document.getElementById('root');

if (!rootElement) {
  throw new Error('Unable to mount Invoke V7: root element was not found.');
}

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>
);
