import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { StarterSourceLinks } from './StarterSourceLinks';

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe('starter source links', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  const render = (urls?: string[]) =>
    act(() =>
      root.render(
        <ChakraProvider value={system}>
          <StarterSourceLinks name="Example" urls={urls} />
        </ChakraProvider>
      )
    );

  it('omits the control when there are no safe sources', async () => {
    await render();
    expect(host.querySelector('a, button')).toBeNull();
    await render(['javascript:alert(1)']);
    expect(host.querySelector('a, button')).toBeNull();
  });

  it('opens a single source with a native new-tab link', async () => {
    await render(['https://example.com/model', 'https://example.com/model']);
    const link = host.querySelector('a');
    expect(link?.href).toBe('https://example.com/model');
    expect(link?.target).toBe('_blank');
    expect(link?.rel).toContain('noreferrer');
    expect(host.querySelector('button')).toBeNull();
  });

  it.each(['pointer', 'keyboard'])(
    'offers source links and restores focus after %s opening and Escape',
    async (input) => {
      await render(['https://example.com/model', 'https://example.org/quantized']);
      const trigger = host.querySelector('button')!;
      if (input === 'pointer') {
        await userEvent.click(trigger);
      } else {
        await userEvent.tab();
        expect(document.activeElement).toBe(trigger);
        await userEvent.keyboard('{ArrowDown}');
      }
      await expect.poll(() => document.querySelectorAll('[role="menuitem"]').length).toBe(2);
      const links = [...document.querySelectorAll<HTMLAnchorElement>('a[role="menuitem"]')];
      expect(links.map((link) => link.href)).toEqual(['https://example.com/model', 'https://example.org/quantized']);
      expect(links.every((link) => link.target === '_blank' && link.rel.includes('noreferrer'))).toBe(true);
      await expect.poll(() => document.activeElement?.getAttribute('role')).toBe('menu');
      await userEvent.keyboard('{Escape}');
      await expect.poll(() => document.activeElement).toBe(trigger);
      await expect.poll(() => trigger.getAttribute('aria-expanded')).toBe('false');
    }
  );
});
