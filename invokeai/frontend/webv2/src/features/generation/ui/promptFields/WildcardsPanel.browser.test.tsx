import type { WildcardCatalog } from '@features/generation/ui/useWildcards';

import { ChakraProvider } from '@chakra-ui/react';
import { WildcardsPanel } from '@features/generation/ui/promptFields/WildcardsPanel';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const catalog: WildcardCatalog = {
  create: vi.fn(),
  isLoading: false,
  knownNames: new Set(['colors']),
  remove: vi.fn(),
  update: vi.fn(),
  wildcards: [{ id: 'w1', name: 'colors', values: ['red', 'green'] }],
};

const openEditor = async (showSyntaxHighlighting: boolean) => {
  host = document.createElement('div');
  host.style.width = '380px';
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <WildcardsPanel catalog={catalog} showSyntaxHighlighting={showSyntaxHighlighting} onInsert={vi.fn()} />
      </ChakraProvider>
    );
  });

  await act(async () => {
    await userEvent.click([...host!.querySelectorAll('button')][0]!);
  });

  const values = host!.querySelector('textarea')!;

  await act(async () => {
    await userEvent.fill(values, '(oil painting)1.3\nwatercolor+\n{ink|charcoal} sketch');
  });
  await act(async () => {
    await new Promise((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(resolve));
    });
  });
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('wildcard values editor', () => {
  it('colours attention and nested dynamic syntax in the values', async () => {
    await openEditor(true);

    const spans = [...host!.querySelectorAll('[aria-hidden="true"] pre span')].map((span) => span.textContent);

    // Attention weighting survives into the generated prompt, and a nested
    // `{a|b}` is live syntax inside a wildcard value, so both are coloured.
    expect(spans).toContain('1.3');
    expect(spans).toContain('+');
    expect(spans).toContain('{');
    expect(spans).toContain('|');
  });

  it('numbers each value in the gutter', async () => {
    await openEditor(true);

    expect([...host!.querySelectorAll('[data-line-number]')].map((entry) => entry.textContent)).toEqual([
      '1',
      '2',
      '3',
    ]);
  });

  it('still numbers the values when highlighting is off', async () => {
    // Line numbers are not syntax colouring, so they do not follow that setting.
    await openEditor(false);

    expect([...host!.querySelectorAll('[data-line-number]')]).toHaveLength(3);
    expect(host!.querySelectorAll('[aria-hidden="true"] pre span')).toHaveLength(0);
  });
});
