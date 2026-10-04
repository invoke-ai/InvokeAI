import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, expect, it } from 'vitest';
import { userEvent } from 'vitest/browser';

import { RemovableTag } from './RemovableTag';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const noop = (): void => undefined;

const renderTag = async (label: string): Promise<HTMLElement> => {
  host = document.createElement('div');
  host.style.cssText = 'display:flex;width:160px;';
  document.body.append(host);
  root = createRoot(host);
  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <RemovableTag removeLabel="Show everyone" onRemove={noop}>
          {label}
        </RemovableTag>
      </ChakraProvider>
    );
  });
  return [...host.querySelectorAll<HTMLElement>('span')].find((node) => node.textContent === label)!;
};

const tooltipWith = (text: string) =>
  [...document.querySelectorAll('[role="tooltip"]')].find((node) => node.textContent?.includes(text));

it('reveals a truncated label in full on hover and keeps the remove control named', async () => {
  const label = 'Account: someone-with-a-very-long-display-name@example.com';
  const element = await renderTag(label);
  expect(element.scrollWidth).toBeGreaterThan(element.clientWidth);
  expect(host!.querySelector('button[aria-label="Show everyone"]')).not.toBeNull();

  await userEvent.hover(element);
  await expect.poll(() => tooltipWith(label)).toBeDefined();
});

it('shows no tooltip for a label that fits', async () => {
  const element = await renderTag('Mine');
  await userEvent.hover(element);
  await new Promise((resolve) => {
    setTimeout(resolve, 800);
  });
  expect(tooltipWith('Mine')).toBeUndefined();
});
