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

import { createSystemPrompt, updateSystemPrompt } from './systemPrompts';

const dto = {
  content: 'Rewrite the prompt.',
  id: 'prompt-1',
  is_public: true,
  max_tokens: 500,
  name: 'Ref2VA',
  user_id: 'system',
};

const sentBody = (): Record<string, unknown> => {
  const init = transport.apiFetchJson.mock.calls[0]?.[1] as RequestInit | undefined;

  return JSON.parse(init?.body as string) as Record<string, unknown>;
};

beforeEach(() => {
  transport.apiFetch.mockReset();
  transport.apiFetchJson.mockReset();
  transport.apiFetchJson.mockResolvedValue(dto);
});

describe('system prompt max_tokens', () => {
  it('maps a stored cap onto the record', async () => {
    const created = await createSystemPrompt({ content: dto.content, maxTokens: 500, name: dto.name });

    expect(created.maxTokens).toBe(500);
  });

  it('reads a prompt with no cap as null', async () => {
    transport.apiFetchJson.mockResolvedValue({ ...dto, max_tokens: null });

    const created = await createSystemPrompt({ content: dto.content, maxTokens: null, name: dto.name });

    expect(created.maxTokens).toBeNull();
  });

  it('sends the cap on create', async () => {
    await createSystemPrompt({ content: dto.content, maxTokens: 500, name: dto.name });

    expect(sentBody().max_tokens).toBe(500);
  });

  it('sends an explicit null on update so a cap can be cleared', async () => {
    // Explicit null clears the value; omission preserves it.
    await updateSystemPrompt('prompt-1', { content: dto.content, maxTokens: null, name: dto.name });

    const body = sentBody();

    expect('max_tokens' in body).toBe(true);
    expect(body.max_tokens).toBeNull();
  });
});
