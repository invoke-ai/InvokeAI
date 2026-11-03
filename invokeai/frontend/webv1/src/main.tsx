import ReactDOM from 'react-dom/client';

import { getDeploymentBaseUrl } from './common/util/baseUrl';
import { retireDefaultServiceWorker } from './common/util/retireDefaultServiceWorker';

const boot = async () => {
  if (await retireDefaultServiceWorker(getDeploymentBaseUrl())) {
    return;
  }
  const { default: InvokeAIUI } = await import('./app/components/InvokeAIUI');
  ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(<InvokeAIUI />);
};

void boot().catch(() => {
  const root = document.getElementById('root');
  if (root) {
    root.textContent = 'Unable to start the legacy frontend. Reload to retry.';
    const retry = document.createElement('button');
    retry.textContent = 'Reload';
    retry.addEventListener('click', () => window.location.reload());
    root.appendChild(retry);
  }
});
