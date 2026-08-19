import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({ apiFetch: vi.fn(), apiFetchJson: vi.fn() }));

vi.mock('@platform/transport/http', () => ({ apiFetch: mocks.apiFetch, apiFetchJson: mocks.apiFetchJson }));

import {
  deleteLibraryWorkflowThumbnail,
  getAllWorkflowTags,
  getWorkflowTagCounts,
  listLibraryWorkflows,
  setLibraryWorkflowThumbnail,
  touchLibraryWorkflowLastRunAt,
} from './api';

describe('workflow library api', () => {
  beforeEach(() => {
    mocks.apiFetch.mockReset().mockResolvedValue(new Response());
    mocks.apiFetchJson.mockReset().mockResolvedValue({});
  });

  it('touches last_run_at with a PUT to the record-scoped endpoint', async () => {
    await touchLibraryWorkflowLastRunAt('abc');

    expect(mocks.apiFetch).toHaveBeenCalledWith('/api/v1/workflows/i/abc/last_run_at', {
      method: 'PUT',
      signal: undefined,
    });
  });

  it('writes a thumbnail as multipart form data under the image field', async () => {
    const blob = new Blob(['fake-image-bytes'], { type: 'image/png' });

    await setLibraryWorkflowThumbnail('abc', blob);

    expect(mocks.apiFetch).toHaveBeenCalledTimes(1);
    const [path, init] = mocks.apiFetch.mock.calls[0] as [string, RequestInit];

    expect(path).toBe('/api/v1/workflows/i/abc/thumbnail');
    expect(init.method).toBe('PUT');
    expect(init.body).toBeInstanceOf(FormData);

    const uploaded = (init.body as FormData).get('image');

    expect(uploaded).toBeInstanceOf(Blob);
    expect((uploaded as Blob).size).toBe(blob.size);
    expect((uploaded as Blob).type).toBe(blob.type);
  });

  it('deletes a thumbnail with a DELETE to the record-scoped endpoint', async () => {
    await deleteLibraryWorkflowThumbnail('abc');

    expect(mocks.apiFetch).toHaveBeenCalledWith('/api/v1/workflows/i/abc/thumbnail', {
      method: 'DELETE',
      signal: undefined,
    });
  });

  it('forwards repeated tags query params when listing workflows', async () => {
    mocks.apiFetchJson.mockResolvedValue({ items: [], page: 1, pages: 1, total: 0 });

    await listLibraryWorkflows({ category: 'user', page: 1, tags: ['upscaling', 'lora'] });

    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);
    const [path] = mocks.apiFetchJson.mock.calls[0] as [string];
    const params = new URL(path, 'http://localhost').searchParams;

    expect(params.getAll('tags')).toEqual(['upscaling', 'lora']);
  });

  it('gets tag counts for the given tags', async () => {
    mocks.apiFetchJson.mockResolvedValue({ lora: 2, upscaling: 1 });

    const result = await getWorkflowTagCounts({ tags: ['lora', 'upscaling'] });

    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);
    const [path] = mocks.apiFetchJson.mock.calls[0] as [string];
    const params = new URL(path, 'http://localhost').searchParams;

    expect(path.startsWith('/api/v1/workflows/counts_by_tag?')).toBe(true);
    expect(params.getAll('tags')).toEqual(['lora', 'upscaling']);
    expect(result).toEqual({ lora: 2, upscaling: 1 });
  });

  it('gets all workflow tags', async () => {
    mocks.apiFetchJson.mockResolvedValue(['lora', 'upscaling']);

    const result = await getAllWorkflowTags();

    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/workflows/tags', { signal: undefined });
    expect(result).toEqual(['lora', 'upscaling']);
  });
});
