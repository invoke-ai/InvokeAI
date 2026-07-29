import { QueryClient } from '@tanstack/react-query';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const transport = vi.hoisted(() => ({
  apiFetch: vi.fn(),
  apiFetchJson: vi.fn(),
}));

vi.mock('@platform/transport/http', () => ({
  absolutizeApiUrl: (path: string) => path,
  apiFetch: transport.apiFetch,
  apiFetchJson: transport.apiFetchJson,
}));

vi.mock('@platform/state/accountLifecycle', () => ({
  assertAccountScopeCurrent: vi.fn(),
  captureAccountScope: () => ({ signal: new AbortController().signal }),
}));

import { fetchPromptTemplateImage, updatePromptTemplate, type PromptTemplateUpdateDraft } from './promptTemplates';

const dto = {
  id: 'template-1',
  image: '/api/v1/style_presets/i/template-1/image',
  is_public: false,
  name: 'Cinematic',
  preset_data: { negative_prompt: '', positive_prompt: '{prompt}, cinematic' },
  type: 'user' as const,
  user_id: 'user-1',
};

const getFormData = (): FormData => {
  const init = transport.apiFetchJson.mock.calls[0]?.[1] as RequestInit | undefined;

  expect(init?.method).toBe('PATCH');
  expect(init?.body).toBeInstanceOf(FormData);
  return init?.body as FormData;
};

const update = async (image: PromptTemplateUpdateDraft['image']): Promise<FormData> => {
  transport.apiFetchJson.mockResolvedValue(dto);

  await updatePromptTemplate('template/1', {
    image,
    name: 'Cinematic',
    negativePrompt: '',
    positivePrompt: '{prompt}, cinematic',
  });

  expect(transport.apiFetchJson.mock.calls[0]?.[0]).toBe('/api/v1/style_presets/i/template%2F1');
  return getFormData();
};

beforeEach(() => {
  transport.apiFetch.mockReset();
  transport.apiFetchJson.mockReset();
});

describe('prompt template image updates', () => {
  it('sends preserve_image without re-uploading an unchanged image', async () => {
    const body = await update({ kind: 'preserve' });

    expect(body.get('preserve_image')).toBe('true');
    expect(body.get('image')).toBeNull();
  });

  it('omits both image fields to explicitly remove an image', async () => {
    const body = await update({ kind: 'remove' });

    expect(body.get('preserve_image')).toBeNull();
    expect(body.get('image')).toBeNull();
  });

  it('uploads the replacement image without preserve_image', async () => {
    const replacement = new Blob(['replacement'], { type: 'image/png' });
    const body = await update({ blob: replacement, kind: 'replace' });

    expect(body.get('preserve_image')).toBeNull();
    const uploaded = body.get('image');
    expect(uploaded).toBeInstanceOf(Blob);
    expect((uploaded as Blob).type).toBe('image/png');
    await expect((uploaded as Blob).text()).resolves.toBe('replacement');
  });
});

it('fetches template images through the authenticated relative-path transport', async () => {
  const image = new Blob(['image'], { type: 'image/png' });
  transport.apiFetch.mockResolvedValue(new Response(image));
  const queryClient = new QueryClient();

  await expect(
    queryClient.fetchQuery({
      queryFn: ({ signal }) => fetchPromptTemplateImage('folder/template', signal),
      queryKey: ['template-image'],
    })
  ).resolves.toEqual(image);

  expect(transport.apiFetch).toHaveBeenCalledWith('/api/v1/style_presets/i/folder%2Ftemplate/image', {
    signal: expect.any(AbortSignal),
  });
});
