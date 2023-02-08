import { store } from 'app/store';
import { persistStore } from 'redux-persist';

export const persistor = persistStore(store);
