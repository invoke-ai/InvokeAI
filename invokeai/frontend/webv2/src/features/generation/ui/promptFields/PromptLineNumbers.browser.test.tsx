import { ChakraProvider } from '@chakra-ui/react';
import { PromptTextarea } from '@features/generation/ui/promptFields/PromptTextarea';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (value: string, widthPx = 320) => {
  host = document.createElement('div');
  host.style.width = `${widthPx}px`;
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <PromptTextarea
          aria-label="Values"
          defaultHeightPx={200}
          minHeightPx={96}
          readOnly
          resizeHandleAriaLabel="Resize"
          showLineNumbers
          showSyntaxHighlighting
          value={value}
        />
      </ChakraProvider>
    );
  });
  // The gutter sizes itself from a measured mirror, so let layout settle.
  await act(async () => {
    await new Promise((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(resolve));
    });
  });
};

const gutterEntries = () => [...host!.querySelectorAll('[data-line-number]')];

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('prompt line numbers', () => {
  it('numbers every logical line', async () => {
    await render('red\ngreen\nblue');

    expect(gutterEntries().map((entry) => entry.textContent)).toEqual(['1', '2', '3']);
  });

  it('numbers a blank line rather than skipping it', async () => {
    await render('red\n\nblue');

    expect(gutterEntries().map((entry) => entry.textContent)).toEqual(['1', '2', '3']);
  });

  it('gives a wrapped line one number and the height of its wrap', async () => {
    const long = 'a '.repeat(120).trim();

    await render(`short\n${long}\nshort`);

    const heights = gutterEntries().map((entry) => entry.getBoundingClientRect().height);

    expect(gutterEntries().map((entry) => entry.textContent)).toEqual(['1', '2', '3']);
    // The wrapped line keeps a single number but occupies several rows, so its
    // gutter entry has to be taller than its unwrapped neighbours.
    expect(heights[1]).toBeGreaterThan(heights[0] * 2);
    expect(heights[2]).toBeCloseTo(heights[0], 1);
  });
});
