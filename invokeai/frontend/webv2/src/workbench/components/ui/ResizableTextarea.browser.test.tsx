import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { PromptTextarea } from '@workbench/widgets/generate/promptFields/PromptTextarea';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ResizableTextarea } from './ResizableTextarea';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const renderTextarea = async (maxHeightPx?: number) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const onResizeEnd = vi.fn();

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <ResizableTextarea
          aria-label="Prompt"
          defaultHeightPx={100}
          maxHeightPx={maxHeightPx}
          minHeightPx={60}
          resizeHandleAriaLabel="Resize prompt"
          onResizeEnd={onResizeEnd}
        />
      </ChakraProvider>
    );
  });

  return {
    handle: host.querySelector<HTMLElement>('[role="separator"]')!,
    onResizeEnd,
    textarea: host.querySelector<HTMLTextAreaElement>('textarea')!,
  };
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('ResizableTextarea', () => {
  it('allows unbounded pointer and keyboard growth while keeping Home at the minimum', async () => {
    const { handle, onResizeEnd, textarea } = await renderTextarea();

    await act(() => {
      handle.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'End' }));
      handle.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowDown' }));
    });
    expect(getComputedStyle(textarea).height).toBe('112px');

    await act(() => {
      handle.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientY: 100 }));
      window.dispatchEvent(new PointerEvent('pointermove', { clientY: 700 }));
      window.dispatchEvent(new PointerEvent('pointerup'));
    });
    expect(getComputedStyle(textarea).height).toBe('712px');
    expect(onResizeEnd).toHaveBeenLastCalledWith(712);

    await act(() => handle.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Home' })));
    expect(getComputedStyle(textarea).height).toBe('60px');
    expect(handle.hasAttribute('aria-valuemax')).toBe(false);
  });

  it('retains bounded End behavior when a maximum is supplied', async () => {
    const { handle, textarea } = await renderTextarea(140);

    await act(() => handle.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'End' })));

    expect(getComputedStyle(textarea).height).toBe('140px');
    expect(handle.getAttribute('aria-valuemax')).toBe('140');
  });

  it('uses the legacy prompt font size for both the textarea and syntax underlay', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <PromptTextarea
            aria-label="Prompt"
            defaultHeightPx={100}
            minHeightPx={60}
            resizeHandleAriaLabel="Resize prompt"
            showSyntaxHighlighting
            value="(sunset:1.2)"
            onChange={vi.fn()}
          />
        </ChakraProvider>
      );
    });

    const textarea = host.querySelector<HTMLTextAreaElement>('textarea')!;
    const underlay = host.querySelector<HTMLElement>('pre')!;

    expect(getComputedStyle(textarea).fontSize).toBe('13.12px');
    expect(getComputedStyle(underlay).fontSize).toBe('13.12px');
  });
});
