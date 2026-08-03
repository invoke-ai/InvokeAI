import { ChakraProvider } from '@chakra-ui/react';
import { parseDateTokens } from '@platform/search/dateTokens';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { GallerySearchField, getGallerySearchSegments } from './GallerySearchField';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const onChange = vi.fn();

const renderField = async (value: string, isInvalid = false) => {
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <GallerySearchField
          ariaLabel="Search gallery items"
          isInvalid={isInvalid}
          placeholder="Search items"
          value={value}
          onChange={onChange}
        />
      </ChakraProvider>
    )
  );
};

const getInput = (): HTMLInputElement => {
  const input = host?.querySelector('input');

  if (!input) {
    throw new Error('search input did not render');
  }

  return input;
};

/** The mirror is the input's own sibling; the search icon is also aria-hidden. */
const getMirror = (): HTMLElement => {
  const mirror = getInput().previousElementSibling;

  if (!(mirror instanceof HTMLElement) || mirror.getAttribute('aria-hidden') !== 'true') {
    throw new Error('mirror layer not found beside the input');
  }

  return mirror;
};

const getChips = (): HTMLElement[] =>
  Array.from(getMirror().querySelectorAll<HTMLElement>('span')).filter(
    (span) => getComputedStyle(span).backgroundColor !== 'rgba(0, 0, 0, 0)'
  );

beforeEach(() => {
  host = document.createElement('div');
  host.style.cssText = 'left:0;position:fixed;top:0;width:320px;';
  document.body.append(host);
  root = createRoot(host);
  onChange.mockClear();
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('getGallerySearchSegments', () => {
  it('reproduces the value exactly, so the mirror stays in register with the input', () => {
    for (const value of ['', 'sunset', 'from:7d sunset', '  from:today  to:2026-01-02 x', 'from:nope tail']) {
      const segments = getGallerySearchSegments(value, parseDateTokens(value));

      expect(segments.map((segment) => segment.text).join(''), value).toBe(value);
    }
  });

  it('chips only the token, never the space in front of it', () => {
    const value = 'a from:7d';
    const segments = getGallerySearchSegments(value, parseDateTokens(value));

    expect(segments.filter((segment) => segment.kind === 'chip').map((segment) => segment.text)).toEqual(['from:7d']);
  });

  it('marks tokens the grammar rejected', () => {
    const value = 'from:nope';
    const [chip] = getGallerySearchSegments(value, parseDateTokens(value)).filter((s) => s.kind === 'chip');

    expect(chip?.isInvalid).toBe(true);
  });
});

describe('GallerySearchField', () => {
  it('renders a chip for the token and leaves the rest plain', async () => {
    await renderField('from:7d sunset');

    expect(getChips().map((chip) => chip.textContent)).toEqual(['from:7d']);
  });

  it('reports invalid input to assistive tech', async () => {
    await renderField('from:nope', true);

    expect(getInput().getAttribute('aria-invalid')).toBe('true');
  });

  it('reports typing back to the owner', async () => {
    await renderField('');
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;

    await act(async () => {
      setter?.call(getInput(), 'from:today');
      getInput().dispatchEvent(new Event('input', { bubbles: true }));
      await Promise.resolve();
    });

    expect(onChange).toHaveBeenCalledWith('from:today');
  });
});
