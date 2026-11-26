import { accountLifecycle } from '@platform/state/accountLifecycle';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  apiFetchJson: vi.fn(),
  apiFetchRaw: vi.fn(),
  sleep: vi.fn(),
}));

vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  apiFetchJson: mocks.apiFetchJson,
  apiFetchRaw: mocks.apiFetchRaw,
  sleep: mocks.sleep,
}));

import { recoverGalleryArchives } from '@features/gallery/archiveRecovery';

import { downloadGalleryArchive, resumeGalleryArchives } from './archives';

describe('downloadGalleryArchive', () => {
  beforeEach(() => {
    accountLifecycle.activate('user-a');
    mocks.apiFetchJson.mockReset();
    mocks.apiFetchRaw.mockReset();
    mocks.sleep.mockReset();
  });

  it('aborts polling before another account can request the prior account archive', async () => {
    let resumeSleep: (() => void) | undefined;
    mocks.apiFetchJson.mockResolvedValue({ bulk_download_item_name: 'user-a.zip' });
    mocks.apiFetchRaw.mockResolvedValue(new Response(null, { status: 404 }));
    mocks.sleep.mockImplementationOnce(
      () =>
        new Promise<void>((resolve) => {
          resumeSleep = resolve;
        })
    );

    const download = downloadGalleryArchive({ imageNames: ['user-a.png'] });

    await vi.waitFor(() => {
      expect(mocks.sleep).toHaveBeenCalledOnce();
    });
    const postSignal = (mocks.apiFetchJson.mock.calls[0]?.[1] as RequestInit | undefined)?.signal;
    const pollSignal = (mocks.apiFetchRaw.mock.calls[0]?.[1] as RequestInit | undefined)?.signal;
    const sleepSignal = mocks.sleep.mock.calls[0]?.[1] as AbortSignal;

    accountLifecycle.invalidate();
    accountLifecycle.activate('user-b');
    resumeSleep?.();

    await expect(download).rejects.toThrow('no longer active');
    expect(postSignal?.aborted).toBe(true);
    expect(pollSignal?.aborted).toBe(true);
    expect(sleepSignal.aborted).toBe(true);
    expect(mocks.apiFetchRaw).toHaveBeenCalledTimes(1);
  });

  it('rejects when the server no longer has the archive', async () => {
    mocks.apiFetchJson.mockResolvedValue({ bulk_download_item_name: 'failed.zip' });
    mocks.apiFetchRaw.mockResolvedValue(new Response(null, { status: 410 }));

    await expect(downloadGalleryArchive({ imageNames: ['a.png'] })).rejects.toThrow('no longer available');
  });

  it('resumes claimed archives once ready and skips ones that are gone', async () => {
    mocks.apiFetchJson.mockResolvedValue({ bulk_download_item_names: ['ready.zip', 'gone.zip'] });
    const responses: Record<string, Response[]> = {
      'gone.zip': [new Response(null, { status: 410 })],
      'ready.zip': [new Response(null, { status: 404 }), new Response('zip-bytes', { status: 200 })],
    };
    mocks.apiFetchRaw.mockImplementation((url: string) => responses[url.split('/').at(-1)!]!.shift());
    mocks.sleep.mockResolvedValue(undefined);
    const archives: { blob: Blob; fileName: string }[] = [];

    await resumeGalleryArchives(accountLifecycle.capture(), (archive) => archives.push(archive));

    expect(mocks.apiFetchJson).toHaveBeenCalledWith(
      '/api/v1/images/download/claim',
      expect.objectContaining({ method: 'POST' })
    );
    expect(archives.map((archive) => archive.fileName)).toEqual(['ready.zip']);
    await expect(archives[0]?.blob.text()).resolves.toBe('zip-bytes');
  });

  it('delivers a fetched archive before a sibling still preparing is cut off', async () => {
    mocks.apiFetchJson.mockResolvedValue({ bulk_download_item_names: ['ready.zip', 'slow.zip'] });
    mocks.apiFetchRaw.mockImplementation((url: string) =>
      Promise.resolve(url.endsWith('ready.zip') ? new Response('zip-bytes') : new Response(null, { status: 404 }))
    );
    mocks.sleep.mockImplementation(
      (_ms: number, signal: AbortSignal) =>
        new Promise((_resolve, reject) => {
          signal.addEventListener('abort', () => reject(signal.reason));
        })
    );
    const delivered: string[] = [];

    const resume = resumeGalleryArchives(accountLifecycle.capture(), ({ fileName }) => delivered.push(fileName));
    await vi.waitFor(() => expect(delivered).toEqual(['ready.zip']));
    accountLifecycle.invalidate();

    await expect(resume).rejects.toThrow();
    expect(delivered).toEqual(['ready.zip']);
  });

  it('claims again after the retry delay and stops when the account ends', async () => {
    mocks.apiFetchJson.mockResolvedValue({ bulk_download_item_names: [] });
    mocks.sleep.mockResolvedValueOnce(undefined);
    const owner = accountLifecycle.capture();

    await recoverGalleryArchives(owner, () => undefined);
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(2);

    mocks.sleep.mockImplementationOnce(() => {
      accountLifecycle.invalidate();
      return Promise.reject(new DOMException('Aborted', 'AbortError'));
    });
    await recoverGalleryArchives(accountLifecycle.activate('user-a'), () => undefined);
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(3);
  });
});
