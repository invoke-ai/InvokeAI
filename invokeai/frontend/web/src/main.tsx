import { AppConfig } from 'app/invokeai';
import ReactDOM from 'react-dom/client';

import Component from './component';

const testConfig: Partial<AppConfig> = {
  disabledTabs: ['nodes'],
  disabledFeatures: ['upscaling'],
};

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <Component config={testConfig} />
);
