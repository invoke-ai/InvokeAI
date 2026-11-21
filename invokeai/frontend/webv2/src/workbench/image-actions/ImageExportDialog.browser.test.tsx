import type { GalleryImage } from '@features/gallery';
import type * as projectsApi from '@workbench/projects/api';
import type { Ref } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { DEFAULT_PREFERENCES, getWorkbenchPreferences, patchWorkbenchPreferences } from '@workbench/settings/store';
import { act, createRef, useImperativeHandle } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { RequestImageExport } from './imageExport';

import { saveImageExport } from './imageExport';
import { useImageExportDialog } from './useImageExportDialog';

const mocks = vi.hoisted(() => ({ downloadBlob: vi.fn(async () => {}) }));

vi.mock('@platform/browser/downloadBlob', () => ({ downloadBlob: mocks.downloadBlob }));
// The preference store is real; only its backend write is stubbed.
vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<typeof projectsApi>()),
  setClientStateValue: async () => {},
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    i18n: { language: 'en' },
    t: (key: string, values?: Record<string, unknown>) => (values ? `${key}:${JSON.stringify(values)}` : key),
  }),
}));

const image: GalleryImage = {
  boardId: 'none',
  height: 768,
  imageCategory: 'general',
  imageName: 'moon.png',
  imageUrl: '/api/v1/images/i/moon.png/full',
  queuedAt: '2026-11-21T00:00:00Z',
  sourceQueueItemId: 'queue-item',
  starred: false,
  thumbnailUrl: '/api/v1/images/i/moon.png/thumbnail',
  width: 1024,
};

/** The dialog's module loads on first open; a cold transform under a busy run can take a while. */
const DIALOG_LOAD_WAIT = { timeout: 10_000 };

type ExportHandle = { requestImageExport: RequestImageExport };

const Probe = ({ ref }: { ref: Ref<ExportHandle> }) => {
  const { dialog, requestImageExport } = useImageExportDialog();

  useImperativeHandle(ref, () => ({ requestImageExport }), [requestImageExport]);

  return dialog;
};

const exportRef = createRef<ExportHandle>();
const encodedBlobs: Blob[] = [];
const exportUrls: string[] = [];
let host: HTMLDivElement | null = null;
let root: Root | null = null;
let queryClient = new QueryClient();
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const respond = (input: RequestInfo | URL, init?: RequestInit): Response => {
  const url = String(input);

  if (url.includes('/export?')) {
    const blob = new Blob([`encoded ${String(exportUrls.length)}`], { type: 'image/webp' });

    exportUrls.push(url.slice(url.indexOf('?') + 1));
    encodedBlobs.push(blob);
    return new Response(blob, { headers: { 'content-type': blob.type } });
  }
  if (url.endsWith('/export/names')) {
    const { format } = JSON.parse(String(init?.body)) as { format: string };

    return Response.json({ file_names: [`moon-night.${format === 'webp' ? 'webp' : 'png'}`] });
  }
  if (init?.method === 'HEAD') {
    return new Response(null, { headers: { 'content-length': '40000' } });
  }
  return new Response(null, { status: 404 });
};

beforeEach(async () => {
  accountLifecycle.activate('image-export-test');
  await patchWorkbenchPreferences(DEFAULT_PREFERENCES);
  exportUrls.length = 0;
  encodedBlobs.length = 0;
  mocks.downloadBlob.mockClear();
  vi.stubGlobal(
    'fetch',
    vi.fn((input: RequestInfo | URL, init?: RequestInit) => Promise.resolve(respond(input, init)))
  );
  queryClient = new QueryClient();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <QueryClientProvider client={queryClient}>
        <ChakraProvider value={system}>
          <Probe ref={exportRef} />
        </ChakraProvider>
      </QueryClientProvider>
    )
  );
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  vi.unstubAllGlobals();
  accountLifecycle.invalidate();
});

describe('image export dialog', () => {
  it('previews what the settings encode and saves those bytes under the server’s name', async () => {
    await act(() =>
      exportRef.current!.requestImageExport({ image, kind: 'image' }, (target, settings) =>
        target.kind === 'image'
          ? saveImageExport({ image: target.image, owner: captureAccountScope(), queryClient, settings })
          : Promise.resolve()
      )
    );
    const dialog = () => document.querySelector<HTMLElement>('[role="dialog"][data-state="open"]');
    const segment = (value: string) =>
      dialog()!.querySelector(`input[type="radio"][value="${value}"]`)!.closest<HTMLElement>('label')!;

    await expect.poll(() => exportUrls, DIALOG_LOAD_WAIT).toEqual(['format=png&metadata=all']);

    await userEvent.click(segment('webp'));
    await userEvent.click(segment('0.5'));
    await expect.poll(() => exportUrls.at(-1)).toBe('format=webp&long_edge=512&quality=90');
    await expect.poll(() => dialog()?.textContent).toContain('moon-night.webp');
    const encodesBeforeSave = exportUrls.length;

    const exportButton = Array.from(dialog()!.querySelectorAll('button')).find(
      (button) => button.textContent === 'widgets.gallery.export.export'
    )!;
    await userEvent.click(exportButton);

    await expect.poll(() => mocks.downloadBlob.mock.calls.length).toBe(1);
    expect(mocks.downloadBlob).toHaveBeenCalledWith(encodedBlobs.at(-1), 'moon-night.webp');
    expect(exportUrls).toHaveLength(encodesBeforeSave);
    await expect.poll(dialog).toBeNull();
    expect(getWorkbenchPreferences().imageExportSettings).toEqual({
      format: 'webp',
      longEdge: 512,
      metadata: 'all',
      namePattern: '{name}',
      quality: 90,
    });
  });
});
