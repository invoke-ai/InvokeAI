import { store } from 'app/store/store';
import { persistStore } from 'redux-persist';

export const persistor = persistStore(store);
