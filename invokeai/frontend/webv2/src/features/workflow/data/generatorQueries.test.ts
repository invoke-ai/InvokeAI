import { galleryItemNamesOptions } from '@features/gallery/queries';
import { QueryClient } from '@tanstack/react-query';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { resolveWorkflowGenerators } from './generatorQueries';

const http = vi.hoisted(() => ({ apiFetchJson: vi.fn() }));

vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  apiFetchJson: http.apiFetchJson,
}));

const promptsRequest = (seed: number | null, combinatorial = false) => ({
  key: 'k',
  nodeId: 'gen',
  nodeLabel: 'Prompts',
  request: { combinatorial, kind: 'dynamicPrompts' as const, maxPrompts: 3, prompt: '{a|b}', seed },
});
const boardRequest = {
  key: 'k',
  nodeId: 'board',
  nodeLabel: 'Board images',
  request: { boardId: 'board-1', category: 'images' as const, kind: 'boardImages' as const },
};
const sentSeeds = () =>
  http.apiFetchJson.mock.calls
    .filter(([path]) => String(path).includes('dynamicprompts'))
    .map(([, init]) => JSON.parse((init as { body: string }).body).seed as number | null);

describe('resolveWorkflowGenerators', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    http.apiFetchJson.mockReset();
    http.apiFetchJson.mockImplementation((path: string) =>
      String(path).includes('dynamicprompts')
        ? Promise.resolve({ error: null, prompts: ['a', 'b'] })
        : Promise.resolve({
            items: [
              { kind: 'image', name: 'a.png' },
              { kind: 'video', name: 'clip.mp4' },
            ],
            total: 2,
          })
    );
  });

  it('draws a fresh seed for an unseeded random prompt on every submission, so repeats differ', async () => {
    const first = await resolveWorkflowGenerators(queryClient, [promptsRequest(null)]);
    const second = await resolveWorkflowGenerators(queryClient, [promptsRequest(null)]);

    expect(first.resolutions).toEqual({ gen: { items: ['a', 'b'], key: 'k' } });
    expect(second.errors).toEqual([]);
    expect(sentSeeds()).toHaveLength(2);
    expect(sentSeeds().every((seed) => Number.isInteger(seed))).toBe(true);
    expect(sentSeeds()[0]).not.toBe(sentSeeds()[1]);

    // A pinned seed reuses its expansion: one request for two submissions.
    await resolveWorkflowGenerators(queryClient, [promptsRequest(7)]);
    await resolveWorkflowGenerators(queryClient, [promptsRequest(7)]);
    expect(sentSeeds()).toEqual([...sentSeeds().slice(0, 2), 7]);
  });

  it('lists a board fresh rather than from the preview cache, and keeps only its images', async () => {
    queryClient.setQueryData(
      galleryItemNamesOptions({ boardId: 'board-1', galleryView: 'images', searchTerm: '' }).queryKey,
      { items: [{ kind: 'image', name: 'stale.png' }], total: 1 }
    );

    const { errors, resolutions } = await resolveWorkflowGenerators(queryClient, [boardRequest]);

    expect(errors).toEqual([]);
    expect(http.apiFetchJson).toHaveBeenCalledTimes(1);
    expect(resolutions).toEqual({ board: { items: [{ image_name: 'a.png' }], key: 'k' } });
  });

  it('names the generator in every failure and resolves nothing for it', async () => {
    http.apiFetchJson.mockImplementation((path: string) =>
      String(path).includes('dynamicprompts')
        ? Promise.resolve({ error: 'Unbalanced braces', prompts: [] })
        : Promise.reject(new Error('offline'))
    );

    const { errors, resolutions } = await resolveWorkflowGenerators(queryClient, [
      promptsRequest(null, true),
      boardRequest,
    ]);

    expect(resolutions).toEqual({});
    expect(errors.sort()).toEqual(['Board images: offline', 'Prompts: Unbalanced braces']);
  });
});
