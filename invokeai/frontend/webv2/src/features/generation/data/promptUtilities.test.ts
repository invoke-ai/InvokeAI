import { describe, expectTypeOf, it } from 'vitest';

import type { ExpandPromptRequest, ExpandPromptResponse } from './promptUtilities';

describe('prompt utility transport contracts', () => {
  it('accepts an optional nullable expansion seed', () => {
    const request = {
      model_key: 'text-model',
      prompt: 'Expand this',
      seed: null,
      task_id: 'task-1',
    } satisfies ExpandPromptRequest;

    expectTypeOf(request.seed).toEqualTypeOf<null>();
  });

  it('requires the backend-selected expansion seed in responses', () => {
    expectTypeOf<ExpandPromptResponse['seed']>().toEqualTypeOf<number>();
  });
});
