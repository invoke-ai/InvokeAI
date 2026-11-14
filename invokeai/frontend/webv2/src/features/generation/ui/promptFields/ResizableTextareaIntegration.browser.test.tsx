import { Box, ChakraProvider } from '@chakra-ui/react';
import { adjustFocusedPromptAttention } from '@features/generation/ui/promptFields/promptAttentionHotkeys';
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

const WELL_HOST_CSS = { '--wb-control-well': '{colors.bg}', '& textarea': { bg: 'var(--wb-control-well)' } };

const renderTextarea = async (maxHeightPx?: number, well?: { zoom: number }) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const onResizeEnd = vi.fn();
  const textarea = (
    <ResizableTextarea
      aria-label="Prompt"
      defaultHeightPx={100}
      maxHeightPx={maxHeightPx}
      minHeightPx={60}
      resizeHandleAriaLabel="Resize prompt"
      onResizeEnd={onResizeEnd}
    />
  );

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        {well ? (
          // A scaled well host, as the node editor is: it paints its controls on the well, and its zoom puts the inner
          // bottom edge inside a device pixel.
          <Box css={WELL_HOST_CSS} data-control-well="" transform={`scale(${well.zoom})`}>
            {textarea}
          </Box>
        ) : (
          textarea
        )}
      </ChakraProvider>
    );
  });

  return {
    fade: host.querySelector<HTMLElement>('[data-resizable-textarea-fade]')!,
    handle: host.querySelector<HTMLElement>('[role="separator"]')!,
    onResizeEnd,
    textarea: host.querySelector<HTMLTextAreaElement>('textarea')!,
  };
};

const lastColor = (gradient: string) => [...gradient.matchAll(/(?:rgba?|oklch)\([^)]*\)/g)].at(-1)?.[0];

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
    });
    expect(document.activeElement).toBe(textarea);
    const accentProbe = document.createElement('div');
    accentProbe.style.border = '1px solid var(--chakra-colors-accent-solid)';
    document.body.append(accentProbe);
    const accentBorderColor = getComputedStyle(accentProbe).borderTopColor;
    accentProbe.remove();
    expect(getComputedStyle(textarea).outlineStyle).toBe('none');
    await expect.poll(() => getComputedStyle(textarea).borderTopColor).toBe(accentBorderColor);

    await act(async () => {
      await userEvent.unhover(textarea);
    });
    await expect.poll(() => getComputedStyle(textarea).borderTopColor).toBe(accentBorderColor);
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
    expect(textarea.getAttribute('role')).toBe('textbox');
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

  it('fades to whatever lies behind it outside a well host', async () => {
    const { fade, textarea } = await renderTextarea();

    expect(getComputedStyle(textarea.parentElement!.parentElement!).maskImage).not.toBe('none');
    expect(getComputedStyle(fade).display).toBe('none');
  });

  it('covers a scaled well host’s bottom edge with an opaque well fade and repaints the border over it', async () => {
    const zoom = 1.37;
    const { fade, textarea } = await renderTextarea(undefined, { zoom });
    const style = getComputedStyle(textarea);
    const lineHeight = Number.parseFloat(style.lineHeight);
    await act(() => {
      textarea.value = Array.from({ length: 20 }, (_, index) => `line ${index}`).join('\n');
      textarea.dispatchEvent(new Event('input', { bubbles: true }));
    });
    // Half a line past a line boundary, so a line straddles the inner bottom edge.
    textarea.scrollTop =
      Math.ceil(textarea.clientHeight / lineHeight) * lineHeight - textarea.clientHeight + lineHeight / 2;
    expect(textarea.scrollTop).toBeGreaterThan(0);

    const box = textarea.getBoundingClientRect();
    const fill = fade.getBoundingClientRect();
    const border = getComputedStyle(fade, '::after');
    const innerBottom = box.bottom - Number.parseFloat(style.borderBottomWidth) * zoom;
    expect(getComputedStyle(textarea.parentElement!.parentElement!).maskImage).toBe('none');
    // The fill is the text area's own bottom, so it covers the inner bottom edge and the row the border shares with it;
    // the repainted border's inner edge is the text area's.
    expect(Math.abs(fill.bottom - box.bottom)).toBeLessThanOrEqual(0.5);
    expect(Math.abs(fill.left - box.left)).toBeLessThanOrEqual(0.5);
    expect(Math.abs(fill.right - box.right)).toBeLessThanOrEqual(0.5);
    expect(
      Math.abs(fill.bottom - Number.parseFloat(border.borderBottomWidth) * zoom - innerBottom)
    ).toBeLessThanOrEqual(0.5);
    expect(lastColor(getComputedStyle(fade).backgroundImage)).toBe(style.backgroundColor);
    expect(style.backgroundColor).not.toMatch(/rgba|\//);
    expect(border.borderBottomLeftRadius).toBe(style.borderBottomLeftRadius);
    expect(border.borderBottomColor).toBe(style.borderBottomColor);
  });

  it('repaints the well fade’s border in the text area’s hover, focus and error colours', async () => {
    const { fade, textarea } = await renderTextarea(undefined, { zoom: 1 });
    const settled = () =>
      act(async () => {
        // Flush style so the transitions exist before waiting for their actual completion.
        textarea.getBoundingClientRect();
        await Promise.all(
          textarea.parentElement!.parentElement!.getAnimations({ subtree: true }).map((animation) => animation.finished)
        );
      });
    const colors = () => [
      getComputedStyle(fade, '::after').borderBottomColor,
      getComputedStyle(textarea).borderBottomColor,
    ];
    const rest = colors()[1];

    await act(() => userEvent.hover(textarea));
    await settled();
    const [hoverRing, hoverBorder] = colors();
    expect(hoverBorder).not.toBe(rest);
    expect(hoverRing).toBe(hoverBorder);

    await act(async () => {
      await userEvent.unhover(textarea);
      await userEvent.tab();
    });
    await settled();
    const [focusRing, focusBorder] = colors();
    expect(focusBorder).not.toBe(rest);
    expect(focusRing).toBe(focusBorder);

    await act(() => textarea.setAttribute('aria-invalid', 'true'));
    await settled();
    const [errorRing, errorBorder] = colors();
    expect(errorBorder).not.toBe(focusBorder);
    expect(errorRing).toBe(errorBorder);
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
      const [value, setValue] = useState('(hello world+)++');
      const handleChange = useCallback((event: ChangeEvent<HTMLTextAreaElement>) => {
        setValue(event.currentTarget.value);
      }, []);

      return (
        <PromptTextarea
          aria-label="Prompt"
          attentionSyntax="edit"
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
    textarea.setSelectionRange(1, 6, 'backward');

    await act(() => {
      expect(adjustFocusedPromptAttention('increment', false)).toBe(true);
    });

    await expect.poll(() => textarea.value).toBe('(hello world)+++');
    expect(textarea.value.slice(textarea.selectionStart, textarea.selectionEnd)).toBe('hello');
    expect(textarea.selectionDirection).toBe('backward');

    await act(() => {
      expect(adjustFocusedPromptAttention('decrement', false)).toBe(true);
    });
    await expect.poll(() => textarea.value).toBe('(hello world+)++');
    expect(textarea.value.slice(textarea.selectionStart, textarea.selectionEnd)).toBe('hello');

    await act(async () => {
      await userEvent.keyboard('{Control>}z{/Control}');
    });
    await expect.poll(() => textarea.value).toBe('(hello world)+++');
  });

  // Synthetic input must not mutate a read-only prompt.
  it('leaves a read-only prompt alone', async () => {
    const ReadOnlyHarness = () => (
      <PromptTextarea
        aria-label="Prompt"
        attentionSyntax="edit"
        defaultHeightPx={96}
        minHeightPx={56}
        readOnly
        resizeHandleAriaLabel="Resize prompt"
        showSyntaxHighlighting
        value="hello world"
        onChange={vi.fn()}
      />
    );

    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <ReadOnlyHarness />
        </ChakraProvider>
      );
    });

    const textarea = host.querySelector<HTMLTextAreaElement>('textarea')!;

    textarea.focus();
    textarea.setSelectionRange(0, 5);

    await act(() => {
      expect(adjustFocusedPromptAttention('increment', false)).toBe(false);
    });

    expect(textarea.value).toBe('hello world');
  });
});
