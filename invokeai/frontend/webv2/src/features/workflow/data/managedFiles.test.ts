import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  deleteManagedFile,
  getManagedFile,
  MANAGED_FILE_ACCEPT,
  MANAGED_FILE_ACCEPTED_EXTENSIONS,
  managedFileQueryOptions,
  uploadManagedFile,
} from './managedFiles';

const httpMock = vi.hoisted(() => ({ apiFetch: vi.fn(), apiFetchJson: vi.fn() }));

vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  apiFetch: httpMock.apiFetch,
  apiFetchJson: httpMock.apiFetchJson,
}));

const FILE_DTO = {
  content_type: 'text/markdown',
  created_at: '2026-06-25T04:53:34.000Z',
  file_id: 'file-1',
  file_name: 'notes.md',
  size_bytes: 2048,
};

const FILE = {
  contentType: 'text/markdown',
  createdAt: '2026-06-25T04:53:34.000Z',
  id: 'file-1',
  name: 'notes.md',
  sizeBytes: 2048,
};

describe('managed files transport', () => {
  beforeEach(() => {
    httpMock.apiFetch.mockReset();
    httpMock.apiFetchJson.mockReset();
  });

  it('maps the wire DTO into the domain shape and resolves the encoded file URL', async () => {
    httpMock.apiFetchJson.mockResolvedValue(FILE_DTO);
    const signal = new AbortController().signal;

    await expect(getManagedFile('file/1', signal)).resolves.toEqual(FILE);
    expect(httpMock.apiFetchJson).toHaveBeenCalledWith('/api/v1/files/i/file%2F1', { signal });
  });

  it('uploads through the multipart "file" field and maps the created file', async () => {
    httpMock.apiFetchJson.mockResolvedValue(FILE_DTO);
    const file = new File(['# Notes'], 'notes.md', { type: 'text/markdown' });
    const signal = new AbortController().signal;

    await expect(uploadManagedFile(file, signal)).resolves.toEqual(FILE);
    const [path, init] = httpMock.apiFetchJson.mock.calls[0]!;

    expect(path).toBe('/api/v1/files/upload');
    expect(init).toEqual({ body: expect.any(FormData), method: 'POST', signal });
    expect((init.body as FormData).get('file')).toBe(file);
  });

  it('deletes by id and accepts the 204 response', async () => {
    httpMock.apiFetch.mockResolvedValue(new Response(null, { status: 204 }));
    const signal = new AbortController().signal;

    await expect(deleteManagedFile('file/1', signal)).resolves.toBeUndefined();
    expect(httpMock.apiFetch).toHaveBeenCalledWith('/api/v1/files/i/file%2F1', { method: 'DELETE', signal });
  });

  it('offers the server-supported extensions and leaves an absent id unfetched', () => {
    expect(MANAGED_FILE_ACCEPTED_EXTENSIONS).toEqual([
      '.csv',
      '.json',
      '.md',
      '.markdown',
      '.pdf',
      '.txt',
      '.yaml',
      '.yml',
    ]);
    expect(MANAGED_FILE_ACCEPT).toBe('.csv,.json,.md,.markdown,.pdf,.txt,.yaml,.yml');
    expect(managedFileQueryOptions('').enabled).toBe(false);

    const options = managedFileQueryOptions('file-1');

    expect(options.enabled).toBe(true);
    expect(options.queryKey).toEqual(['workflow', 'managed-file', 'file-1']);
    expect(options.retry).toBe(false);
    expect(options.staleTime).toBeGreaterThanOrEqual(60_000);
  });
});
