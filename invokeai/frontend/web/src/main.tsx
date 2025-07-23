import type { StorageDriverApi } from 'app/store/enhancers/reduxRemember/driver';
import ReactDOM from 'react-dom/client';

import InvokeAIUI from './app/components/InvokeAIUI';

let state: Record<string, any> = {};
const storageDriverApi: StorageDriverApi = {
  getItem: (key: string) => {
    return Promise.resolve(state[key]);
  },
  setItem: (key: string, value: any) => {
    state[key] = value;
    return Promise.resolve(value);
  },
  clear: () => {
    state = {};
    return Promise.resolve();
  },
};

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <InvokeAIUI storageDriverApi={storageDriverApi} />
);
