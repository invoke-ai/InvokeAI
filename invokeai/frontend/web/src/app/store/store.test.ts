import { Buffer } from 'node:buffer';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import {
  externalTokenAdopted,
  logout,
  sessionExpiredLogout,
  setCredentials,
  staleCredentialsDiscarded,
} from 'features/auth/store/authSlice';
import { isModalOpenChanged, videosToChangeSelected } from 'features/changeBoardModal/store/slice';
import { bboxHeightChanged, bboxWidthChanged, canvasUndo } from 'features/controlLayers/store/canvasSlice';
import { positivePromptChanged } from 'features/controlLayers/store/paramsSlice';
import { refImageAdded } from 'features/controlLayers/store/refImagesSlice';
import { deleteVideosWithDialog } from 'features/deleteVideoModal/store/state';
import {
  $gallerySelection,
  markNextSelectionAutoSwitched,
  resetGallerySelectionSource,
} from 'features/gallery/store/gallerySelectionSource';
import {
  autoAddBoardIdChanged,
  boardIdSelected,
  imageSelected,
  selectionChanged,
} from 'features/gallery/store/gallerySlice';
import { undo as nodesUndo, workflowNameChanged } from 'features/nodes/store/nodesSlice';
import { upscaleInitialImageChanged } from 'features/parameters/store/upscaleSlice';
import { appInfoApi } from 'services/api/endpoints/appInfo';
import type { S } from 'services/api/types';
import { beforeAll, describe, expect, it, vi } from 'vitest';

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

// The canvas undo filter throttles rapid same-type actions with a `window.setTimeout` reset
// timer, and this file runs in the node environment. Only that one API is needed.
beforeAll(() => {
  vi.stubGlobal('window', { setTimeout });
});

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
    ['a deliberate logout', () => logout()],
    ['a cross-tab account switch', () => externalTokenAdopted(tokenFor('other-user'))],
  ])('clears the workspace slices and their undo stacks on %s', (_label, makeAction) => {
    // Both account-change paths — a same-tab logout and another tab's foreign token — must
    // wipe the workspace: it is personal state, and it is where deleted-image references live
    // (raster/control layers, node image fields, reference images), so a cross-user batch
    // delete aborted mid-run must not leave the next account holding references to images
    // that no longer exist. The undo assertions are the half that is easy to lose: the
    // undoable filters keep cross-slice actions out of history without emptying it, so a
    // reset alone leaves the previous account's states one ctrl+Z away. The adoption case
    // must work at the reducer level — the synthetic logout it triggers never reaches
    // listeners, so a listener-based clear would pass the logout case and fail this one.
    const store = createStore();
    store.dispatch(setCredentials({ token: tokenFor(user.user_id), user }));
    const initialCanvas = store.getState().canvas.present;
    const initialNodes = store.getState().nodes.present;
    store.dispatch(bboxWidthChanged({ width: initialCanvas.bbox.rect.width + 64 }));
    store.dispatch(bboxHeightChanged({ height: initialCanvas.bbox.rect.height + 64 }));
    store.dispatch(workflowNameChanged('previous user workflow'));
    store.dispatch(refImageAdded());
    store.dispatch(upscaleInitialImageChanged({ image_name: 'previous-user.png', width: 64, height: 64 }));
    expect(store.getState().canvas.present).not.toEqual(initialCanvas);
    expect(store.getState().nodes.present).not.toEqual(initialNodes);
    expect(store.getState().refImages.entities).toHaveLength(1);

    store.dispatch(makeAction());

    expect(store.getState().canvas.present).toEqual(initialCanvas);
    expect(store.getState().nodes.present).toEqual(initialNodes);
    expect(store.getState().refImages.entities).toHaveLength(0);
    expect(store.getState().upscale.upscaleInitialImage).toBeNull();
    // The stacks are asserted directly as well as behaviorally: with few seeded actions an
    // undo's target can coincide with the initial state, and the behavioral check alone would
    // stay green with the clear missing.
    expect(store.getState().canvas.past).toHaveLength(0);
    expect(store.getState().nodes.past).toHaveLength(0);
    store.dispatch(canvasUndo());
    store.dispatch(nodesUndo());
    expect(store.getState().canvas.present).toEqual(initialCanvas);
    expect(store.getState().nodes.present).toEqual(initialNodes);
    // The clears must land *after* the reset pass, and this is the assertion that pins the
    // order: clears that run first leave the filtered reset as redux-undo's _latestUnfiltered,
    // so the next account's first action pushes the previous account's state into past — one
    // action and one ctrl+Z resurrect it. Dispatch as the new account, undo, and the state
    // must come back to the *reset*, not to what was seeded above.
    store.dispatch(bboxWidthChanged({ width: initialCanvas.bbox.rect.width + 128 }));
    store.dispatch(workflowNameChanged('next user workflow'));
    store.dispatch(canvasUndo());
    store.dispatch(nodesUndo());
    expect(store.getState().canvas.present).toEqual(initialCanvas);
    expect(store.getState().nodes.present).toEqual(initialNodes);
  });

  it.each([
    ['the session merely expires', () => sessionExpiredLogout()],
    ['another tab refreshes the same user token', () => externalTokenAdopted(tokenFor(user.user_id))],
    ['stale multiuser credentials are discarded on a single-user switch', () => staleCredentialsDiscarded()],
  ])('keeps the whole workspace when %s', (_label, makeAction) => {
    // None of these is an account change. A timeout's user is coming back, and wiping hours of
    // canvas or workflow work over it would be destructive — deleted-image references under
    // expiry are handled by the batch loops resolving with partial data so `handleDeletions`
    // can prune. A same-user token refresh changes nothing at all. And the single-user mode
    // switch keeps the same human at the machine — worse, in single-user mode the
    // unauthenticated persist is accepted, so a wipe there would overwrite the stored
    // workspace for good. Asserted across every purged slice, not just one: each slice decides
    // independently which actions it resets on, so a single-slice probe cannot see one of the
    // others going over-eager.
    const store = createStore();
    store.dispatch(setCredentials({ token: tokenFor(user.user_id), user }));
    store.dispatch(workflowNameChanged('my unsaved workflow'));
    store.dispatch(bboxWidthChanged({ width: 1024 }));
    store.dispatch(refImageAdded());
    store.dispatch(upscaleInitialImageChanged({ image_name: 'mine.png', width: 64, height: 64 }));

    store.dispatch(makeAction());

    expect(store.getState().nodes.present.name).toBe('my unsaved workflow');
    expect(store.getState().canvas.present.bbox.rect.width).toBe(1024);
    expect(store.getState().refImages.entities).toHaveLength(1);
    expect(store.getState().upscale.upscaleInitialImage?.image_name).toBe('mine.png');
  });

  it('discards stale credentials on a mode switch without the account-change action', () => {
    // A source guard in the manner of ChangeBoardModal.test.ts: the store tests above exercise
    // the actions, but nothing else pins WHICH action the mode-switch branch dispatches.
    // Reverting it to `logout()` re-arms the account-change wipe on a path where the same
    // human keeps the machine — and where single-user mode persists the wipe over their stored
    // workspace. `logout()` belongs to UserMenu alone.
    const source = readFileSync(
      fileURLToPath(new URL('../../features/auth/components/ProtectedRoute.tsx', import.meta.url)),
      'utf8'
    );
    expect(source).toContain('dispatch(staleCredentialsDiscarded())');
    expect(source).not.toContain('dispatch(logout())');
  });

  it('still clears credentials and the api cache when stale multiuser credentials are discarded', async () => {
    const store = createStore();
    store.dispatch(setCredentials({ token: tokenFor(user.user_id), user }));
    await store.dispatch(appInfoApi.util.upsertQueryData('getRuntimeConfig', undefined, runtimeConfig));

    store.dispatch(staleCredentialsDiscarded());

    expect(store.getState().auth.token).toBeNull();
    expect(store.getState().auth.isAuthenticated).toBe(false);
    // The cache was fetched under multiuser visibility scoping, so it does not carry over.
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
      operation_id: 2,
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
  it('publishes selections through the real store wiring', () => {
    // The per-listener tests build their own store, so nothing else fails if the registration in
    // store.ts is deleted — and without it no selection is ever published: the auto-switch mark is
    // never spent, and the viewer's reveal machine never hears about any selection at all. This is
    // the one test that dispatches through createStore()'s own listeners.
    const store = createStore();
    resetGallerySelectionSource(); // module singleton; start from empty

    markNextSelectionAutoSwitched();
    store.dispatch(imageSelected('auto-switched.png'));
    expect($gallerySelection.get()).toMatchObject({ name: 'auto-switched.png', isAutoSwitch: true });

    store.dispatch(imageSelected('user-clicked.png'));
    expect($gallerySelection.get()).toMatchObject({ name: 'user-clicked.png', isAutoSwitch: false });
  });
});
