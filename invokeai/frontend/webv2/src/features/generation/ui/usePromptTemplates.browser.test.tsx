import type { PromptTemplateRecord } from '@features/generation/data/promptTemplates';

import { promptTemplateKeys } from '@features/generation/data/promptTemplates';
import { usePromptTemplates, type PromptTemplateCatalog } from '@features/generation/ui/usePromptTemplates';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, createRef, type Ref, useImperativeHandle } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const transport = vi.hoisted(() => ({
  apiFetch: vi.fn(),
  apiFetchJson: vi.fn(),
}));

vi.mock('@platform/transport/http', () => ({
  apiFetch: transport.apiFetch,
  apiFetchJson: transport.apiFetchJson,
}));

vi.mock('@platform/state/accountLifecycle', () => ({
  assertAccountScopeCurrent: vi.fn(),
  captureAccountScope: () => ({ signal: new AbortController().signal }),
}));

vi.mock('@features/generation/ui/GenerationUiContext', () => ({
  useGenerationUi: () => ({ account: { currentUserId: 'user-1', multiuserEnabled: true } }),
}));

const dto = {
  id: 'template-1',
  image: null,
  is_public: false,
  name: 'Cinematic',
  preset_data: { negative_prompt: '', positive_prompt: '{prompt}, cinematic' },
  type: 'user' as const,
  user_id: 'user-1',
};

const template: PromptTemplateRecord = {
  hasImage: true,
  id: dto.id,
  isDefault: false,
  isPublic: false,
  name: dto.name,
  negativePrompt: dto.preset_data.negative_prompt,
  positivePrompt: dto.preset_data.positive_prompt,
  userId: dto.user_id,
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let queryClient: QueryClient;
const catalogRef = createRef<PromptTemplateCatalog>();
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const Probe = ({ ref }: { ref: Ref<PromptTemplateCatalog> }) => {
  const value = usePromptTemplates({ isEnabled: false });

  useImperativeHandle(ref, () => value, [value]);

  return null;
};

beforeEach(async () => {
  vi.clearAllMocks();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  queryClient = new QueryClient();

  await act(() => {
    root?.render(
      <QueryClientProvider client={queryClient}>
        <Probe ref={catalogRef} />
      </QueryClientProvider>
    );
  });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('prompt template mutation invalidation', () => {
  it('invalidates the template list exactly once after creation', async () => {
    transport.apiFetchJson.mockResolvedValue(dto);
    const invalidateQueries = vi.spyOn(queryClient, 'invalidateQueries');

    await act(async () => {
      await catalogRef.current!.create({
        image: null,
        name: dto.name,
        negativePrompt: '',
        positivePrompt: dto.preset_data.positive_prompt,
      });
    });

    expect(invalidateQueries).toHaveBeenCalledTimes(1);
    expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: promptTemplateKeys.list(), exact: true });
  });

  it.each([
    [
      'update',
      (value: ReturnType<typeof usePromptTemplates>) =>
        value.update(template, {
          image: { kind: 'preserve' },
          name: dto.name,
          negativePrompt: '',
          positivePrompt: dto.preset_data.positive_prompt,
        }),
    ],
    ['delete', (value: ReturnType<typeof usePromptTemplates>) => value.remove(template)],
  ])('invalidates the list and affected image exactly once after %s', async (_operation, run) => {
    transport.apiFetchJson.mockResolvedValue(dto);
    transport.apiFetch.mockResolvedValue(new Response());
    const invalidateQueries = vi.spyOn(queryClient, 'invalidateQueries');

    await act(async () => {
      await run(catalogRef.current!);
    });

    expect(invalidateQueries).toHaveBeenCalledTimes(2);
    expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: promptTemplateKeys.list(), exact: true });
    expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: promptTemplateKeys.image(template.id), exact: true });
  });

  it('invalidates the template list exactly once after import', async () => {
    transport.apiFetch.mockResolvedValue(new Response());
    const invalidateQueries = vi.spyOn(queryClient, 'invalidateQueries');

    await act(async () => {
      await catalogRef.current!.importFile(new File(['name,prompt'], 'templates.csv', { type: 'text/csv' }));
    });

    expect(invalidateQueries).toHaveBeenCalledTimes(1);
    expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: promptTemplateKeys.list(), exact: true });
  });
});
