import { logout, sessionExpiredLogout } from 'features/auth/store/authSlice';
import { isModalOpenChanged, videosToChangeSelected } from 'features/changeBoardModal/store/slice';
import { cancelDeletion, deleteVideosWithDialog } from 'features/deleteVideoModal/store/state';
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

  it.each([
    ['logout', logout],
    ['session expiry', sessionExpiredLogout],
  ])('clears video change-board state on %s', (_label, logOut) => {
    const store = createStore();
    store.dispatch(videosToChangeSelected(['previous-user-video.mp4']));
    store.dispatch(isModalOpenChanged(true));

    store.dispatch(logOut());

    expect(store.getState().changeBoardModal).toMatchObject({
      isModalOpen: false,
      image_names: [],
      video_names: [],
    });
  });

  it.each([
    ['logout', logout],
    ['session expiry', sessionExpiredLogout],
  ])('settles a pending video deletion dialog on %s', async (_label, logOut) => {
    const store = createStore();
    let settled = false;
    const pending = deleteVideosWithDialog(['previous-user-video.mp4'], store).then(
      () => {
        settled = true;
      },
      () => {
        settled = true;
      }
    );

    store.dispatch(logOut());
    await Promise.resolve();
    const settledOnLogout = settled;

    // Keep this proof test isolated even while the production behavior is broken.
    cancelDeletion();
    await pending;

    expect(settledOnLogout).toBe(true);
  });
});
