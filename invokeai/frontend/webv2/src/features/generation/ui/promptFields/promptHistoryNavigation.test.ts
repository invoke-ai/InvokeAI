import type * as accountLifecycleModule from '@platform/state/accountLifecycle';

import { getDefaultGenerateSettings } from '@features/generation/settings';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type * as navigationModule from './promptHistoryNavigation';

vi.mock('./promptFocus', () => ({ isPositivePromptFocused: () => true }));

let account: typeof accountLifecycleModule;
let navigation: typeof navigationModule;

const model = { base: 'sdxl', key: 'model', name: 'Model', type: 'main' } as const;

beforeEach(async () => {
  vi.resetModules();
  navigation = await import('./promptHistoryNavigation');
  account = await import('@platform/state/accountLifecycle');
  account.accountLifecycle.activate('prompt-history-a');
});

afterEach(() => {
  account.accountLifecycle.invalidate();
});

describe('prompt history account ownership', () => {
  it('does not restore account A stashed prompts after account B becomes active', () => {
    const patchValues = vi.fn();
    const values = {
      ...getDefaultGenerateSettings(model),
      negativePrompt: 'account A negative',
      negativePromptEnabled: true,
      positivePrompt: 'account A draft',
    };

    navigation.promptHistoryNavigation.navigate({
      direction: -1,
      models: [model],
      patchValues,
      projectId: 'shared-project-id',
      promptHistory: [{ negativePrompt: null, positivePrompt: 'account A history' }],
      values,
    });
    expect(patchValues).toHaveBeenCalledOnce();

    account.accountLifecycle.activate('prompt-history-b');
    patchValues.mockClear();
    navigation.promptHistoryNavigation.navigate({
      direction: 1,
      models: [model],
      patchValues,
      projectId: 'shared-project-id',
      promptHistory: [{ negativePrompt: null, positivePrompt: 'account B history' }],
      values: { ...values, positivePrompt: 'account B draft' },
    });

    expect(patchValues).not.toHaveBeenCalled();
  });
});
