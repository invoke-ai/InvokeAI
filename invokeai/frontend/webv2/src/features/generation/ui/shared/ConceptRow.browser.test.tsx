import type { GenerateLora } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { type ConceptModelIdentity, ConceptRow, type ConceptRowProps } from './ConceptRow';

const PIXEL =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M/wHwAF/gL+XcX1WQAAAABJRU5ErkJggg==';
const IDENTITY: ConceptModelIdentity = {
  getBaseColorPalette: () => 'gray',
  getBaseLabel: (base) => base,
  getImageUrl: () => PIXEL,
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const makeLora = (model: Partial<GenerateLora['model']> = {}): GenerateLora => ({
  isEnabled: true,
  model: { base: 'sdxl', key: 'lora-1', name: 'Ink Wash', type: 'lora', ...model },
  weight: 0.75,
});

const render = async (lora: GenerateLora, handlers: Partial<Pick<ConceptRowProps, 'onRemove' | 'onUpdate'>> = {}) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <ConceptRow
          identity={IDENTITY}
          lora={lora}
          onRemove={handlers.onRemove ?? vi.fn()}
          onUpdate={handlers.onUpdate ?? vi.fn()}
        />
      </ChakraProvider>
    )
  );

  return host;
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('ConceptRow', () => {
  it('shows the model cover as its thumbnail', async () => {
    const row = await render(makeLora({ cover_image: 'cover.png' }));

    await expect.poll(() => row.querySelector('img')?.getAttribute('data-state')).toBe('visible');
    expect(row.querySelector('img')?.getAttribute('src')).toBe(PIXEL);
  });

  it('falls back to a model icon without a cover', async () => {
    const row = await render(makeLora());

    expect(row.querySelector('img')).toBeNull();
    expect(row.querySelector('.lucide-box')).not.toBeNull();
  });

  it('reports toggles, weight steps, and removal against the model key', async () => {
    const onUpdate = vi.fn();
    const onRemove = vi.fn();
    const row = await render(makeLora(), { onRemove, onUpdate });

    await act(() => userEvent.click(row.querySelector<HTMLElement>('[data-part="control"]')!));
    expect(onUpdate).toHaveBeenLastCalledWith('lora-1', { isEnabled: false });

    row.querySelector<HTMLElement>('[role="slider"]')!.focus();
    await act(() => userEvent.keyboard('{ArrowRight}'));
    expect(onUpdate).toHaveBeenLastCalledWith('lora-1', { weight: 0.8 });

    await act(() =>
      userEvent.click(row.querySelector<HTMLElement>('[aria-label="widgets.generate.removeConceptNamed"]')!)
    );
    expect(onRemove).toHaveBeenCalledWith('lora-1');
  });
});
