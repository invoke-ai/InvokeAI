import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { TriggerPhrasesEditor } from './TriggerPhrasesEditor';

const api = vi.hoisted(() => ({ updateModel: vi.fn() }));
const store = vi.hoisted(() => ({ replaceModelInStore: vi.fn() }));

vi.mock('@features/models/data/api', () => api);
vi.mock('@features/models/data/modelsStore', () => store);
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe('TriggerPhrasesEditor', () => {
  let host: HTMLDivElement;
  let root: Root;
  const onError = vi.fn();
  const mount = (phrases: readonly string[]) =>
    act(() => {
      root.render(
        <ChakraProvider value={system}>
          <TriggerPhrasesEditor modelKey="lora-1" phrases={phrases} onError={onError} />
        </ChakraProvider>
      );
    });
  const lastSaved = () => {
    const [, patch] = api.updateModel.mock.lastCall as [string, { trigger_phrases: string[] }];

    return patch.trigger_phrases;
  };
  const tagTexts = () =>
    [...host.querySelectorAll<HTMLElement>('[data-scope="tags-input"][data-part="item-text"]')].map(
      (element) => element.textContent
    );

  beforeEach(() => {
    api.updateModel.mockReset();
    api.updateModel.mockImplementation((_key: string, patch: { trigger_phrases: string[] }) =>
      Promise.resolve({ key: 'lora-1', trigger_phrases: patch.trigger_phrases })
    );
    store.replaceModelInStore.mockReset();
    onError.mockReset();
    accountLifecycle.activate('trigger-test-a', ':user:trigger-test-a');
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('adds with Enter, refuses a duplicate, and removes from the tag', async () => {
    await mount(['alpha']);
    const input = page.getByRole('textbox', { name: 'models.triggerPhrases' });

    await input.fill('beta');
    await userEvent.keyboard('{Enter}');
    await expect.poll(() => api.updateModel.mock.calls.length).toBe(1);
    expect(lastSaved()).toEqual(['alpha', 'beta']);

    await input.fill('ALPHA');
    await userEvent.keyboard('{Enter}');
    await expect.element(page.getByText('models.triggerPhraseDuplicate')).toBeVisible();
    expect(api.updateModel).toHaveBeenCalledTimes(1);

    await page.getByRole('button', { name: 'models.removeTriggerPhrase' }).first().click();
    await expect.poll(() => api.updateModel.mock.calls.length).toBe(2);
    expect(lastSaved()).toEqual(['beta']);
  });

  it('edits a tag in place on double-click, keeping its position', async () => {
    await mount(['alpha', 'beta', 'gamma']);

    await page.getByText('beta', { exact: true }).dblClick();
    const itemInput = () =>
      host
        .querySelectorAll<HTMLElement>('[data-scope="tags-input"][data-part="item"]')[1]!
        .querySelector<HTMLInputElement>('[data-part="item-input"]')!;
    await expect.poll(() => itemInput().value).toBe('beta');
    expect(document.activeElement).toBe(itemInput());

    await userEvent.keyboard('{Backspace}{Backspace}{Backspace}{Backspace}Beta{Enter}');
    await expect.poll(() => api.updateModel.mock.calls.length).toBe(1);
    expect(lastSaved()).toEqual(['alpha', 'Beta', 'gamma']);
    expect(tagTexts()).toEqual(['alpha', 'Beta', 'gamma']);
  });

  it('puts the list back when a save fails', async () => {
    api.updateModel.mockRejectedValueOnce(new Error('offline'));
    await mount(['alpha']);

    await page.getByRole('textbox', { name: 'models.triggerPhrases' }).fill('beta');
    await userEvent.keyboard('{Enter}');
    await expect.poll(() => onError.mock.calls.length).toBe(1);

    expect(onError).toHaveBeenCalledWith('offline');
    expect(tagTexts()).toEqual(['alpha']);
  });
});
