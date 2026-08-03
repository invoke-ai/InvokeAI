import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { GallerySplitHandle } from './GallerySplitHandle';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const onCommit = vi.fn();
const onPreview = vi.fn();

const renderHandle = async (props: Partial<Parameters<typeof GallerySplitHandle>[0]> = {}) => {
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <GallerySplitHandle
          label="Resize board panel"
          max={600}
          min={120}
          orientation="horizontal"
          sizePx={280}
          onCommit={onCommit}
          onPreview={onPreview}
          {...props}
        />
      </ChakraProvider>
    )
  );

  const separator = document.querySelector('[role="separator"]');

  if (!separator) {
    throw new Error('split handle did not render');
  }

  return separator;
};

const interact = async (run: () => void) => {
  await act(async () => {
    run();
    await Promise.resolve();
  });
};

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  onCommit.mockClear();
  onPreview.mockClear();
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('GallerySplitHandle', () => {
  it('exposes the resize range to assistive tech', async () => {
    const separator = await renderHandle();

    expect(separator.getAttribute('aria-label')).toBe('Resize board panel');
    expect(separator.getAttribute('aria-orientation')).toBe('horizontal');
    expect(separator.getAttribute('aria-valuenow')).toBe('280');
    expect(separator.getAttribute('aria-valuemin')).toBe('120');
    expect(separator.getAttribute('aria-valuemax')).toBe('600');
    expect(separator.getAttribute('tabindex')).toBe('0');
  });

  it('previews during a drag and commits once on release', async () => {
    const separator = await renderHandle();

    await interact(() => {
      separator.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: 0, clientY: 100 }));
    });
    await interact(() => {
      window.dispatchEvent(new PointerEvent('pointermove', { clientX: 0, clientY: 140 }));
      window.dispatchEvent(new PointerEvent('pointermove', { clientX: 0, clientY: 160 }));
    });

    expect(onPreview).toHaveBeenCalledWith(320);
    expect(onPreview).toHaveBeenLastCalledWith(340);
    expect(onCommit).not.toHaveBeenCalled();

    await interact(() => window.dispatchEvent(new PointerEvent('pointerup', { clientX: 0, clientY: 160 })));

    expect(onPreview).toHaveBeenLastCalledWith(null);
    expect(onCommit).toHaveBeenCalledExactlyOnceWith(340);
  });

  it('clamps a drag to the bounds rather than reporting an impossible size', async () => {
    const separator = await renderHandle();

    await interact(() => {
      separator.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: 0, clientY: 100 }));
    });
    await interact(() => window.dispatchEvent(new PointerEvent('pointermove', { clientX: 0, clientY: -9000 })));
    expect(onPreview).toHaveBeenLastCalledWith(120);

    await interact(() => window.dispatchEvent(new PointerEvent('pointermove', { clientX: 0, clientY: 9000 })));
    expect(onPreview).toHaveBeenLastCalledWith(600);
  });

  it('resizes by keyboard along the axis it controls', async () => {
    const separator = await renderHandle();

    await interact(() => separator.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowDown' })));
    expect(onCommit).toHaveBeenLastCalledWith(296);

    await interact(() =>
      separator.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowUp', shiftKey: true }))
    );
    expect(onCommit).toHaveBeenLastCalledWith(248);

    await interact(() => separator.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Home' })));
    expect(onCommit).toHaveBeenLastCalledWith(120);

    await interact(() => separator.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'End' })));
    expect(onCommit).toHaveBeenLastCalledWith(600);
  });
});
