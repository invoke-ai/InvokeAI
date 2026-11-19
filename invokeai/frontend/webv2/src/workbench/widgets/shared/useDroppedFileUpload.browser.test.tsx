import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act, createRef, useImperativeHandle, type Ref } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';

const harness = vi.hoisted(() => ({ store: null as ReturnType<typeof createWorkbenchStore> | null }));

vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useWorkbenchCommands: () => harness.store!.commands,
  useWorkbenchQueries: () => harness.store!.queries,
}));

import { useDroppedFileUpload } from './useDroppedFileUpload';

const i18n = createInstance();
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeAll(async () => {
  const translation = await fetch('/locales/en.json').then((response) => response.json());
  await i18n.use(initReactI18next).init({ initAsync: false, lng: 'en', resources: { en: { translation } } });
});

afterEach(async () => {
  await act(() => root?.unmount());
  root = null;
  harness.store = null;
});

const renderStart = async () => {
  harness.store = createWorkbenchStore(createInitialWorkbenchState());
  const startRef = createRef<ReturnType<typeof useDroppedFileUpload>>();
  const Probe = ({ ref }: { ref: Ref<ReturnType<typeof useDroppedFileUpload>> }) => {
    const start = useDroppedFileUpload({
      pendingKey: 'widgets.preview.dropPending',
      surface: 'preview',
      uploadingKey: 'widgets.preview.dropUploading',
    });
    useImperativeHandle(ref, () => start, [start]);
    return null;
  };
  root = createRoot(document.createElement('div'));
  const render = (key: number) =>
    act(() =>
      root!.render(
        <I18nextProvider i18n={i18n}>
          <Probe key={key} ref={startRef} />
        </I18nextProvider>
      )
    );
  let mounts = 0;
  await render(mounts);
  return {
    remount: () => render(++mounts),
    start: () => startRef.current!,
    store: harness.store,
  };
};

describe('useDroppedFileUpload', () => {
  it('says the upload did not start when its code cannot load, in its own words', async () => {
    const { start, store } = await renderStart();
    const run = vi.fn();

    await act(() => {
      start()([new File(['x'], 'one.png', { type: 'image/png' })], {
        load: () => Promise.reject(new TypeError('Failed to fetch dynamically imported module: /assets/upload.js')),
        run,
        runFailedMessage: 'unused',
      });
    });

    await vi.waitFor(() =>
      expect(store.getState().notifications[0]).toMatchObject({ kind: 'error', title: 'No files uploaded. 1 failed.' })
    );
    expect(JSON.stringify(store.getState().notifications[0])).not.toContain('dynamically imported module');
    expect(run).not.toHaveBeenCalled();
  });

  it('reports code that fails to load at once like any other, and takes a new drop after it', async () => {
    const { start, store } = await renderStart();
    const drop = (load: () => Promise<object>) =>
      act(() => {
        start()([new File(['x'], 'one.png', { type: 'image/png' })], { load, run, runFailedMessage: 'unused' });
      });
    const run = vi.fn(() => Promise.resolve());

    await drop(() => {
      throw new Error('import refused');
    });
    await vi.waitFor(() =>
      expect(store.getState().notifications[0]).toMatchObject({ kind: 'error', title: 'No files uploaded. 1 failed.' })
    );

    await drop(() => Promise.resolve({}));
    await vi.waitFor(() => expect(run).toHaveBeenCalledTimes(1));
  });

  it('names a failure of the work itself with the caller’s message, and takes a new drop once it ended', async () => {
    const { start, store } = await renderStart();
    const drop = (run: () => Promise<void>) =>
      act(() => {
        start()([new File(['x'], 'one.png', { type: 'image/png' })], {
          load: () => Promise.resolve({}),
          run,
          runFailedMessage: "Couldn't add images to canvas",
        });
      });

    await drop(() => Promise.reject(new Error('commit refused')));
    await vi.waitFor(() =>
      expect(store.getState().notifications[0]).toMatchObject({
        kind: 'error',
        title: "Couldn't add images to canvas",
      })
    );

    const run = vi.fn(() => Promise.resolve());
    await drop(run);
    await vi.waitFor(() => expect(run).toHaveBeenCalledTimes(1));
  });

  it('holds each project to one upload at a time, however the user moves between projects', async () => {
    const { remount, start, store } = await renderStart();
    const started: { finish: () => void; projectId: string }[] = [];
    const settle = () =>
      new Promise<void>((resolve) => {
        setTimeout(resolve, 0);
      });
    const pendingNotices = () =>
      store
        .getState()
        .notifications.filter(
          (notice) => notice.title === 'The files you dropped are still uploading. Try again when they are done.'
        ).length;
    const drop = () =>
      act(async () => {
        start()([new File(['x'], 'one.png', { type: 'image/png' })], {
          load: () => Promise.resolve({}),
          run: (_module, { projectId }) =>
            new Promise<void>((resolve) => {
              started.push({ finish: resolve, projectId });
            }),
          runFailedMessage: 'unused',
        });
        await settle();
      });
    const finish = (index: number) =>
      act(async () => {
        started[index]!.finish();
        await settle();
      });
    const projectA = store.getState().activeProjectId;

    await drop();
    let projectB = '';
    await act(() => {
      projectB = store.commands.projects.create().id;
    });
    await remount();
    await drop();
    await act(() => {
      store.commands.projects.switchTo(projectA);
    });
    await drop();

    expect(started.map(({ projectId }) => projectId)).toEqual([projectA, projectB]);
    expect(pendingNotices()).toBe(1);

    await finish(0);
    await drop();
    await act(() => {
      store.commands.projects.switchTo(projectB);
    });
    await drop();
    expect(started.map(({ projectId }) => projectId)).toEqual([projectA, projectB, projectA]);
    expect(pendingNotices()).toBe(2);

    await finish(1);
    await drop();
    await act(() => {
      store.commands.projects.switchTo(projectA);
    });
    await drop();
    expect(started.map(({ projectId }) => projectId)).toEqual([projectA, projectB, projectA, projectB]);
    expect(pendingNotices()).toBe(3);

    await finish(2);
    await finish(3);
  });
});
