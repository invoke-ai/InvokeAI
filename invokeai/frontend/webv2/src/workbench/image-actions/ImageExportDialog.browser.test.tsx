import type { GalleryImage } from '@features/gallery';
import type * as projectsApi from '@workbench/projects/api';
import type { Ref } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { DEFAULT_PREFERENCES, patchWorkbenchPreferences } from '@workbench/settings/store';
import { act, createRef, useImperativeHandle } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { ExecuteImageExport, RequestImageExport } from './imageExport';

import { saveImageExport } from './imageExport';
import { useImageExportDialog } from './useImageExportDialog';

const mocks = vi.hoisted(() => ({ downloadBlob: vi.fn(async (_blob: Blob, _name: string) => {}) }));

vi.mock('@platform/browser/downloadBlob', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  downloadBlob: mocks.downloadBlob,
}));
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

const DIALOG_LOAD_WAIT = { timeout: 10_000 };

type ExportHandle = { requestImageExport: RequestImageExport };

const Probe = ({ ref }: { ref: Ref<ExportHandle> }) => {
  const { dialog, requestImageExport } = useImageExportDialog();

  useImperativeHandle(ref, () => ({ requestImageExport }), [requestImageExport]);

  return dialog;
};

const exportRef = createRef<ExportHandle>();
const exportUrls: string[] = [];
let holdEncodes = false;
let host: HTMLDivElement | null = null;
let root: Root | null = null;
let queryClient = new QueryClient();
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const respond = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
  const url = String(input);

  if (url.includes('/export?')) {
    const body = `encoded ${String(exportUrls.length)}`;

    exportUrls.push(url.slice(url.indexOf('?') + 1));
    if (holdEncodes) {
      await new Promise((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
      });
    }
    return new Response(body, { headers: { 'content-type': 'image/webp' } });
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
  holdEncodes = false;
  mocks.downloadBlob.mockClear();
  vi.stubGlobal('fetch', vi.fn(respond));
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

const saveOneImage: ExecuteImageExport = async (target, settings, options) => {
  if (target.kind === 'image') {
    await saveImageExport({ image: target.image, owner: captureAccountScope(), queryClient, settings, ...options });
  }
};
const dialog = () => document.querySelector<HTMLElement>('[role="dialog"][data-state="open"]');
const exportButton = () =>
  Array.from(dialog()?.querySelectorAll('button') ?? []).find(
    (button) => button.textContent === 'widgets.gallery.export.export'
  );

describe('image export dialog', () => {
  it("previews what the settings encode and saves those bytes under the server's name", async () => {
    await act(() => exportRef.current!.requestImageExport({ image, kind: 'image' }, saveOneImage));
    const segment = (value: string) =>
      dialog()!.querySelector(`input[type="radio"][value="${value}"]`)!.closest<HTMLElement>('label')!;

    await expect.poll(() => exportUrls, DIALOG_LOAD_WAIT).toEqual(['format=png&metadata=all']);

    await userEvent.click(segment('webp'));
    await expect.poll(() => exportUrls.at(-1)).toBe('format=webp&quality=90');
    await expect.poll(() => dialog()?.textContent).toContain('moon-night.webp');
    const encodesBeforeSave = exportUrls.length;

    await userEvent.click(exportButton()!);

    await expect.poll(() => mocks.downloadBlob.mock.calls.length).toBe(1);
    const [savedBlob, savedName] = mocks.downloadBlob.mock.calls[0]!;
    expect(savedName).toBe('moon-night.webp');
    expect(await savedBlob.text()).toBe(`encoded ${String(encodesBeforeSave - 1)}`);
    expect(exportUrls).toHaveLength(encodesBeforeSave);
  });

  it('closes on an account change, open or mid-save, and saves nothing', async () => {
    holdEncodes = true;
    await act(() => exportRef.current!.requestImageExport({ image, kind: 'image' }, saveOneImage));
    await expect.poll(() => exportUrls.length, DIALOG_LOAD_WAIT).toBe(1);

    await act(() => accountLifecycle.invalidate());
    await expect.poll(dialog).toBeNull();

    accountLifecycle.activate('image-export-test-next');
    await act(() => exportRef.current!.requestImageExport({ image, kind: 'image' }, saveOneImage));
    await expect.poll(exportButton).toBeTruthy();
    await userEvent.click(exportButton()!);
    await expect.poll(() => exportUrls.length).toBe(3);

    await act(() => accountLifecycle.invalidate());
    await expect.poll(dialog).toBeNull();
    expect(mocks.downloadBlob).not.toHaveBeenCalled();
  });
});
