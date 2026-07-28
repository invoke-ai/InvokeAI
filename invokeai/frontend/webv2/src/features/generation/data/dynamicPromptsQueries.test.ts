import { QueryClient } from '@tanstack/react-query';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const parseDynamicPrompts = vi.hoisted(() => vi.fn());

vi.mock('@features/generation/data/promptUtilities', () => ({ parseDynamicPrompts }));

import { dynamicPromptsKeys, resolveDynamicPrompts } from './dynamicPromptsQueries';

const request = { combinatorial: true, max_prompts: 100, prompt: 'a __colors__ ball', seed: null };

let queryClient: QueryClient;

beforeEach(() => {
  queryClient = new QueryClient();
  parseDynamicPrompts.mockReset();
  parseDynamicPrompts.mockResolvedValue({ error: null, prompts: ['a red ball'] });
});

describe('resolveDynamicPrompts', () => {
  it('serves a repeat request from the cache', async () => {
    await resolveDynamicPrompts(queryClient, request);
    await resolveDynamicPrompts(queryClient, request);

    expect(parseDynamicPrompts).toHaveBeenCalledTimes(1);
  });

  // Regression: `ensureQueryData` hands back cached data whenever it exists,
  // invalidated or not, so editing a wildcard left the submit path enqueueing an
  // expansion built from the old values.
  it('re-expands after the wildcard catalog is invalidated', async () => {
    await expect(resolveDynamicPrompts(queryClient, request)).resolves.toMatchObject({ prompts: ['a red ball'] });

    parseDynamicPrompts.mockResolvedValue({ error: null, prompts: ['a blue ball'] });
    await queryClient.invalidateQueries({ queryKey: dynamicPromptsKeys.all });

    await expect(resolveDynamicPrompts(queryClient, request)).resolves.toMatchObject({ prompts: ['a blue ball'] });
    expect(parseDynamicPrompts).toHaveBeenCalledTimes(2);
  });

  it('keys separate prompts apart', async () => {
    await resolveDynamicPrompts(queryClient, request);
    await resolveDynamicPrompts(queryClient, { ...request, prompt: 'a __shades__ ball' });

    expect(parseDynamicPrompts).toHaveBeenCalledTimes(2);
  });
});
