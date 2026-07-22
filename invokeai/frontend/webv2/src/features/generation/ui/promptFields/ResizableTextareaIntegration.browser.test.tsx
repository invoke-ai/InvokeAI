import { ChakraProvider } from '@chakra-ui/react';
import {
  adjustFocusedPromptAttention,
  PROMPT_ATTENTION_TARGET_PROPS,
} from '@features/generation/ui/promptFields/promptAttentionHotkeys';
import { PromptTextarea } from '@features/generation/ui/promptFields/PromptTextarea';
import { ResizableTextarea } from '@platform/ui/ResizableTextarea';
import { system } from '@theme/system';
import { act, useCallback, useState, type ChangeEvent } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

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
  it('keeps its focus border while hovered', async () => {
    const { textarea } = await renderTextarea();

    await act(async () => {
      await userEvent.hover(textarea);
      await userEvent.tab();
      await new Promise((resolve) => {
        globalThis.setTimeout(resolve, 200);
      });
    });
    expect(document.activeElement).toBe(textarea);
    const hoveredFocusBorderColor = getComputedStyle(textarea).borderTopColor;
    const accentProbe = document.createElement('div');
    accentProbe.style.border = '1px solid var(--chakra-colors-accent-solid)';
    document.body.append(accentProbe);
    const accentBorderColor = getComputedStyle(accentProbe).borderTopColor;
    accentProbe.remove();
    expect(getComputedStyle(textarea).outlineStyle).toBe('none');
    expect(hoveredFocusBorderColor).toBe(accentBorderColor);

    await act(async () => {
      await userEvent.unhover(textarea);
      await new Promise((resolve) => {
        globalThis.setTimeout(resolve, 200);
      });
    });
    expect(getComputedStyle(textarea).borderTopColor).toBe(accentBorderColor);
  });

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

  it('scrolls through the ScrollArea viewport instead of a native scrollbar', async () => {
    const { textarea } = await renderTextarea(140);

    expect(textarea.dataset.part).toBe('viewport');
    expect(getComputedStyle(textarea).scrollbarWidth).toBe('none');

    await act(() => {
      textarea.value = Array.from({ length: 50 }, (_, index) => `line ${index}`).join('\n');
      textarea.dispatchEvent(new Event('input', { bubbles: true }));
      textarea.scrollTop = 100;
    });

    const scrollbar = host!.querySelector<HTMLElement>('[data-part="scrollbar"]')!;

    await expect.poll(() => scrollbar.hasAttribute('data-overflow-y')).toBe(true);
    await expect.poll(() => host!.querySelector<HTMLElement>('[data-part="thumb"]')!.clientHeight).toBeGreaterThan(0);
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

  it('participates in the shared Ctrl+Up/Down prompt-attention system', async () => {
    const PromptHarness = () => {
      const [value, setValue] = useState('hello world');
      const handleChange = useCallback((event: ChangeEvent<HTMLTextAreaElement>) => {
        setValue(event.currentTarget.value);
      }, []);

      return (
        <PromptTextarea
          {...PROMPT_ATTENTION_TARGET_PROPS}
          aria-label="Prompt"
          defaultHeightPx={96}
          minHeightPx={56}
          resizeHandleAriaLabel="Resize prompt"
          showSyntaxHighlighting
          value={value}
          onChange={handleChange}
        />
      );
    };

    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <PromptHarness />
        </ChakraProvider>
      );
    });

    const textarea = host.querySelector<HTMLTextAreaElement>('textarea')!;

    textarea.focus();
    textarea.setSelectionRange(0, 5);

    await act(() => {
      expect(adjustFocusedPromptAttention('increment', false)).toBe(true);
    });

    await expect.poll(() => textarea.value).toBe('hello+ world');
  });
});
