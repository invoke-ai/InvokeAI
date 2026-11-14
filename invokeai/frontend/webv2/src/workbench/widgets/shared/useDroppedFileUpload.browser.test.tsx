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
      uploadingKey: 'widgets.preview.dropUploading',
    });
    useImperativeHandle(ref, () => start, [start]);
    return null;
  };
  root = createRoot(document.createElement('div'));
  await act(() =>
    root!.render(
      <I18nextProvider i18n={i18n}>
        <Probe ref={startRef} />
      </I18nextProvider>
    )
  );
  return { start: () => startRef.current!, store: harness.store };
};

describe('useDroppedFileUpload', () => {
  it('says the upload did not start when its code cannot load, in its own words', async () => {
    const { start, store } = await renderStart();
    const run = vi.fn();

    await act(() => {
      start()([new File(['x'], 'one.png', { type: 'image/png' })], {
        // What a lost connection or a replaced deploy gives a lazy import.
        load: () => Promise.reject(new TypeError('Failed to fetch dynamically imported module: /assets/upload.js')),
        run,
        runFailedMessage: 'unused',
      });
    });

    await vi.waitFor(() =>
      expect(store.getState().notifications[0]).toMatchObject({ kind: 'error', title: 'No files uploaded. 1 failed.' })
    );
    // The browser's own wording stays out of the notice.
    expect(JSON.stringify(store.getState().notifications[0])).not.toContain('dynamically imported module');
    expect(run).not.toHaveBeenCalled();
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
});
