import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({ apiFetch: vi.fn(), apiFetchJson: vi.fn() }));

vi.mock('@platform/transport/http', () => ({ apiFetch: mocks.apiFetch, apiFetchJson: mocks.apiFetchJson }));

import { getAllWorkflowTags, getLibraryWorkflowRecord, getWorkflowTagCounts, listLibraryWorkflows } from './api';

describe('workflow library api', () => {
  beforeEach(() => {
    mocks.apiFetch.mockReset().mockResolvedValue(new Response());
    mocks.apiFetchJson.mockReset().mockResolvedValue({});
  });

  it('forwards repeated tags query params when listing workflows', async () => {
    mocks.apiFetchJson.mockResolvedValue({ items: [], page: 1, pages: 1, total: 0 });

    await listLibraryWorkflows({ category: 'user', page: 1, tags: ['upscaling', 'lora'] });

    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);
    const [path] = mocks.apiFetchJson.mock.calls[0] as [string];
    const params = new URL(path, 'http://localhost').searchParams;

    expect(params.getAll('tags')).toEqual(['upscaling', 'lora']);
  });

  it('forwards callable picker filters and multiple categories', async () => {
    mocks.apiFetchJson.mockResolvedValue({ items: [], page: 0, pages: 1, total: 0 });

    await listLibraryWorkflows({
      categories: ['user', 'default'],
      callable: true,
      direction: 'ASC',
      isPublic: true,
      orderBy: 'name',
      page: 0,
      query: 'landscape',
    });

    const [path] = mocks.apiFetchJson.mock.calls[0] as [string];
    const params = new URL(path, 'http://localhost').searchParams;

    expect(params.getAll('categories')).toEqual(['user', 'default']);
    expect(params.get('callable')).toBe('true');
    expect(params.get('is_public')).toBe('true');
    expect(params.get('order_by')).toBe('name');
    expect(params.get('direction')).toBe('ASC');
    expect(params.get('query')).toBe('landscape');
  });

  it('fetches a full selected-workflow record for dynamic field resolution', async () => {
    const record = { workflow_id: 'workflow-1', name: 'Workflow 1', workflow: {} };
    mocks.apiFetchJson.mockResolvedValue(record);

    await expect(getLibraryWorkflowRecord('workflow-1')).resolves.toEqual(record);
    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/workflows/i/workflow-1', { signal: undefined });
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
