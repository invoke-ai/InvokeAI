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

  it('fades to whatever lies behind it outside a well host', async () => {
    const { fade, textarea } = await renderTextarea();

    expect(getComputedStyle(textarea.parentElement!.parentElement!).maskImage).not.toBe('none');
    expect(getComputedStyle(fade).display).toBe('none');
  });

  it('retains bounded End behavior when a maximum is supplied', async () => {
    const { handle, textarea } = await renderTextarea(140);

    await act(() => handle.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'End' })));

    expect(getComputedStyle(textarea).height).toBe('140px');
    expect(handle.getAttribute('aria-valuemax')).toBe('140');
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
