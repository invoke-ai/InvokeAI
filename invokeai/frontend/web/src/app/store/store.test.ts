import { Buffer } from 'node:buffer';

import { externalTokenAdopted, logout, sessionExpiredLogout, setCredentials } from 'features/auth/store/authSlice';
import { isModalOpenChanged, videosToChangeSelected } from 'features/changeBoardModal/store/slice';
import { bboxHeightChanged, bboxWidthChanged, canvasUndo } from 'features/controlLayers/store/canvasSlice';
import { positivePromptChanged } from 'features/controlLayers/store/paramsSlice';
import { refImageAdded } from 'features/controlLayers/store/refImagesSlice';
import { deleteVideosWithDialog } from 'features/deleteVideoModal/store/state';
import { autoAddBoardIdChanged, boardIdSelected, selectionChanged } from 'features/gallery/store/gallerySlice';
import { undo as nodesUndo, workflowNameChanged } from 'features/nodes/store/nodesSlice';
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
    // Alternating types on purpose: consecutive same-type canvas actions engage the undo
    // filter's rapid-action throttle, whose reset timer needs `window` — absent in this node
    // test environment. Alternation also holds across it.each cases, which share that
    // module-level throttle state.
    store.dispatch(bboxWidthChanged({ width: initialCanvas.bbox.rect.width + 64 }));
    store.dispatch(bboxHeightChanged({ height: initialCanvas.bbox.rect.height + 64 }));
    store.dispatch(workflowNameChanged('previous user workflow'));
    store.dispatch(refImageAdded());
    expect(store.getState().canvas.present).not.toEqual(initialCanvas);
    expect(store.getState().nodes.present).not.toEqual(initialNodes);
    expect(store.getState().refImages.entities).toHaveLength(1);

    store.dispatch(makeAction());

    expect(store.getState().canvas.present).toEqual(initialCanvas);
    expect(store.getState().nodes.present).toEqual(initialNodes);
    expect(store.getState().refImages.entities).toHaveLength(0);
    // The stacks are asserted directly as well as behaviorally: with few seeded actions an
    // undo's target can coincide with the initial state, and the behavioral check alone would
    // stay green with the clear missing.
    expect(store.getState().canvas.past).toHaveLength(0);
    expect(store.getState().nodes.past).toHaveLength(0);
    store.dispatch(canvasUndo());
    store.dispatch(nodesUndo());
    expect(store.getState().canvas.present).toEqual(initialCanvas);
    expect(store.getState().nodes.present).toEqual(initialNodes);
  });

  it('keeps the workspace when the session merely expires', () => {
    // A timeout is not an account change: the same user is coming back, and wiping hours of
    // canvas or workflow work over it would be destructive. Deleted-image references under
    // expiry are handled elsewhere — the batch loops resolve with partial data on expiry
    // precisely so `handleDeletions` can prune them.
    const store = createStore();
    store.dispatch(setCredentials({ token: tokenFor(user.user_id), user }));
    store.dispatch(workflowNameChanged('my unsaved workflow'));

    store.dispatch(sessionExpiredLogout());

    expect(store.getState().nodes.present.name).toBe('my unsaved workflow');
  });

  it('keeps the workspace when another tab refreshes the same user token', () => {
    const store = createStore();
    store.dispatch(setCredentials({ token: tokenFor(user.user_id), user }));
    store.dispatch(workflowNameChanged('my unsaved workflow'));

    store.dispatch(externalTokenAdopted(tokenFor(user.user_id)));

    expect(store.getState().nodes.present.name).toBe('my unsaved workflow');
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
