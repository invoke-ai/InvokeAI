/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

const ensureModelsLoaded = vi.fn();
const openModelManager = vi.fn();

vi.mock('@features/generation/ui/GenerationUiContext', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useGenerationUi: () => ({
    models: { catalog: [], ensureLoaded: ensureModelsLoaded, openManager: openModelManager },
  }),
}));

vi.mock('@features/generation/ui/useWildcards', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useWildcards: () => ({ wildcards: [] }),
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) =>
      ({
        'widgets.generate.noPromptTriggersAvailable': 'No prompt triggers available',
        'widgets.generate.openModelManager': 'Open model manager',
        'widgets.generate.promptTriggerOptions': 'Prompt trigger options',
        'widgets.generate.searchPromptTriggers': 'Search prompt triggers',
      })[key] ?? key,
  }),
}));

import { PromptTriggerPopover } from './PositivePromptActions';

const POSITIONING = {
  getAnchorRect: () => ({ height: 32, width: 32, x: 100, y: 100 }),
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('PromptTriggerPopover', () => {
  it('keeps empty-state copy and its action together', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(async () => {
      root?.render(
        <ChakraProvider value={system}>
          <PromptTriggerPopover
            loras={[]}
            open
            positioning={POSITIONING}
            selectedModel={undefined}
            onClose={vi.fn()}
            onSelect={vi.fn()}
          />
        </ChakraProvider>
      );
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 50);
      });
    });

    const message = [...document.querySelectorAll<HTMLElement>('p')].find(
      (element) => element.textContent === 'No prompt triggers available'
    );
    const action = [...document.querySelectorAll<HTMLButtonElement>('button')].find(
      (element) => element.textContent === 'Open model manager'
    );

    if (!message || !action) {
      throw new Error('prompt-trigger empty state did not render');
    }

    // With nothing to search or scroll, the popover hugs its empty state: no
    // disabled search input, no fixed-height list region holding space open.
    expect(document.querySelector('[aria-label="Prompt trigger options"]')).toBeNull();
    expect(document.querySelector('input')).toBeNull();

    const messageBounds = message.getBoundingClientRect();
    const actionBounds = action.getBoundingClientRect();

    expect(Math.abs(messageBounds.left - actionBounds.left)).toBeLessThanOrEqual(1);
    expect(actionBounds.top - messageBounds.bottom).toBeLessThanOrEqual(8);
  });
});
