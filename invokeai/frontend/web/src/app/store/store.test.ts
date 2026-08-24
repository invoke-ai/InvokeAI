import { Buffer } from 'node:buffer';

import { externalTokenAdopted, logout, sessionExpiredLogout, setCredentials } from 'features/auth/store/authSlice';
import { isModalOpenChanged, videosToChangeSelected } from 'features/changeBoardModal/store/slice';
import { positivePromptChanged } from 'features/controlLayers/store/paramsSlice';
import { deleteVideosWithDialog } from 'features/deleteVideoModal/store/state';
import { autoSwitchedImages } from 'features/gallery/store/autoSwitchedImages';
import {
  autoAddBoardIdChanged,
  boardIdSelected,
  imageSelected,
  selectionChanged,
} from 'features/gallery/store/gallerySlice';
import { appInfoApi } from 'services/api/endpoints/appInfo';
import type { S } from 'services/api/types';
import { describe, expect, it } from 'vitest';

import { createStore } from './store';

const runtimeConfig = {
  set_fields: ['models_dir'],
  config: { models_dir: '/operator-only/models' },
} as S['InvokeAIAppConfigWithSetFields'];

const user = {
  user_id: 'user',
  email: 'user@example.com',
  display_name: null,
  is_admin: false,
  is_active: true,
};

const tokenFor = (userId: string) =>
  `header.${Buffer.from(JSON.stringify({ user_id: userId })).toString('base64url')}.signature`;

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

  it('clears API data when another tab changes users', async () => {
    const store = createStore();

    store.dispatch(setCredentials({ token: tokenFor(user.user_id), user }));
    await store.dispatch(appInfoApi.util.upsertQueryData('getRuntimeConfig', undefined, runtimeConfig));
    store.dispatch(externalTokenAdopted(tokenFor('other-user')));

    expect(appInfoApi.endpoints.getRuntimeConfig.select()(store.getState()).data).toBeUndefined();
  });

  it('retains API data when another tab refreshes the same user token', async () => {
    const store = createStore();

    store.dispatch(setCredentials({ token: tokenFor(user.user_id), user }));
    await store.dispatch(appInfoApi.util.upsertQueryData('getRuntimeConfig', undefined, runtimeConfig));
    store.dispatch(externalTokenAdopted(tokenFor(user.user_id)));

    expect(appInfoApi.endpoints.getRuntimeConfig.select()(store.getState()).data).toEqual(runtimeConfig);
  });

  it('clears gallery ownership state on a cross-tab account switch', () => {
    const store = createStore();
    store.dispatch(boardIdSelected({ boardId: 'previous-user-private-board' }));
    store.dispatch(autoAddBoardIdChanged('previous-user-private-board'));
    store.dispatch(selectionChanged(['previous-user-video.mp4']));

    store.dispatch(externalTokenAdopted(tokenFor('other-user')));

    expect(store.getState().gallery).toMatchObject({
      selectedBoardId: 'none',
      autoAddBoardId: 'none',
      selection: [],
    });
  });

  it('clears private workspace state on a cross-tab account switch', () => {
    const store = createStore();
    store.dispatch(setCredentials({ token: tokenFor(user.user_id), user }));
    store.dispatch(positivePromptChanged('previous user private prompt'));

    store.dispatch(externalTokenAdopted(tokenFor('other-user')));

    expect(store.getState().params.positivePrompt).toBe('');
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
    const pending = deleteVideosWithDialog(['previous-user-video.mp4'], store);

    store.dispatch(logOut());

    await expect(pending).rejects.toBe('User canceled');
  });
});

describe('gallery listener registration', () => {
  it('settles the auto-switch marker through the real store wiring', () => {
    // The per-listener tests build their own store, so nothing else fails if the registration in
    // store.ts is deleted — and without it the marker never settles, every stale marker suppresses
    // the user's next click on that item, and the exact dead click the marker exists to prevent
    // comes back. This is the one test that dispatches through createStore()'s own listeners.
    const store = createStore();
    autoSwitchedImages.settle(null); // module singleton; start from empty

    autoSwitchedImages.record('auto-switched.png');
    store.dispatch(imageSelected('auto-switched.png'));
    store.dispatch(imageSelected('user-clicked-elsewhere.png'));

    expect(autoSwitchedImages.consume('auto-switched.png'), 'the marker must not survive the selection moving on').toBe(
      false
    );
  });
});
