import ReactDOM from 'react-dom/client';

import InvokeAIUI from './app/components/InvokeAIUI';

const urlParams = new URLSearchParams(window.location.search);
const projectId = urlParams.get('projectId') || '';

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <InvokeAIUI token="INSERT_TOKEN_HERE" projectId={projectId} />
);
