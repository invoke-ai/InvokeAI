import { logout, sessionExpiredLogout } from 'features/auth/store/authSlice';
import { appInfoApi } from 'services/api/endpoints/appInfo';
import type { S } from 'services/api/types';
import { describe, expect, it } from 'vitest';

import { createStore } from './store';

const runtimeConfig = {
  set_fields: ['models_dir'],
  config: { models_dir: '/operator-only/models' },
} as S['InvokeAIAppConfigWithSetFields'];

describe('auth cache isolation', () => {
  it.each([
    ['logout', logout],
    ['session expiry', sessionExpiredLogout],
  ])('clears API data on %s', async (_label, logOut) => {
    const store = createStore();

    await store.dispatch(appInfoApi.util.upsertQueryData('getRuntimeConfig', undefined, runtimeConfig));
    expect(appInfoApi.endpoints.getRuntimeConfig.select()(store.getState()).data).toEqual(runtimeConfig);

    store.dispatch(logOut());

    expect(appInfoApi.endpoints.getRuntimeConfig.select()(store.getState()).data).toBeUndefined();
  });
});
