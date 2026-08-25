import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, useCallback, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { PaintSizeOpacityControls } from './BrushOptions';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe('paint size control', () => {
  let container: HTMLDivElement | null = null;
  let root: Root | null = null;

  afterEach(async () => {
    await act(() => root?.unmount());
    container?.remove();
    container = null;
    root = null;
  });

  it('announces the actual logarithmic brush size', async () => {
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <PaintSizeOpacityControls
            opacity={1}
            setOpacity={vi.fn()}
            setSize={vi.fn()}
            size={0.1}
            sizeLabel="Brush size"
          />
        </ChakraProvider>
      );
    });

    await expect.element(page.getByRole('slider', { name: 'Brush size' })).toHaveAttribute('aria-valuetext', '0.1px');
  });

  it('announces an arbitrary numeric size without snapping its accessible value', async () => {
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <PaintSizeOpacityControls
            opacity={1}
            setOpacity={vi.fn()}
            setSize={vi.fn()}
            size={0.25}
            sizeLabel="Brush size"
          />
        </ChakraProvider>
      );
    });

    await expect.element(page.getByRole('slider', { name: 'Brush size' })).toHaveAttribute('aria-valuetext', '0.25px');
  });

  it('advances from the minimum with the keyboard', async () => {
    const Harness = () => {
      const [size, setSize] = useState(0.1);
      return (
        <PaintSizeOpacityControls
          opacity={1}
          setOpacity={vi.fn()}
          setSize={setSize}
          size={size}
          sizeLabel="Brush size"
        />
      );
    };
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <Harness />
        </ChakraProvider>
      );
    });

    const slider = page.getByRole('slider', { name: 'Brush size' });
    await act(async () => {
      await userEvent.click(slider);
      await userEvent.keyboard('{ArrowRight}');
    });

    await expect.element(page.getByRole('spinbutton', { name: 'Brush size' })).toHaveValue('0.11');
  });

  it('lets a decimal be typed before committing the size', async () => {
    const setSize = vi.fn();
    const Harness = () => {
      const [size, setHarnessSize] = useState(1);
      const handleSize = useCallback((next: number) => {
        setSize(next);
        setHarnessSize(next);
      }, []);
      return (
        <PaintSizeOpacityControls
          opacity={1}
          setOpacity={vi.fn()}
          setSize={handleSize}
          size={size}
          sizeLabel="Brush size"
        />
      );
    };
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <Harness />
        </ChakraProvider>
      );
    });

    const input = page.getByRole('spinbutton', { name: 'Brush size' });
    await act(async () => {
      await userEvent.clear(input);
      await userEvent.fill(input, '0.25');
      await userEvent.keyboard('{Enter}');
    });

    await expect.element(input).toHaveValue('0.25');
    expect(setSize).toHaveBeenLastCalledWith(0.25);
  });

  it('canonicalizes a valid over-precision commit even when the engine value is unchanged', async () => {
    const setSize = vi.fn();
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <PaintSizeOpacityControls
            opacity={1}
            setOpacity={vi.fn()}
            setSize={setSize}
            size={0.25}
            sizeLabel="Brush size"
          />
        </ChakraProvider>
      );
    });

    const input = page.getByRole('spinbutton', { name: 'Brush size' });
    await act(async () => {
      await userEvent.clear(input);
      await userEvent.fill(input, '0.254');
      await userEvent.keyboard('{Enter}');
    });

    expect(setSize).toHaveBeenLastCalledWith(0.25);
    await expect.element(page.getByRole('spinbutton', { name: 'Brush size' })).toHaveValue('0.25');
  });

  it('resets an empty numeric field to the current size on commit', async () => {
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <PaintSizeOpacityControls
            opacity={1}
            setOpacity={vi.fn()}
            setSize={vi.fn()}
            size={5}
            sizeLabel="Brush size"
          />
        </ChakraProvider>
      );
    });

    const input = page.getByRole('spinbutton', { name: 'Brush size' });
    await act(async () => {
      await userEvent.clear(input);
      await userEvent.keyboard('{Enter}');
    });

    await expect.element(page.getByRole('spinbutton', { name: 'Brush size' })).toHaveValue('5');
  });

  it('reaches the exact slider maximum with the End key', async () => {
    const Harness = () => {
      const [size, setSize] = useState(50);
      return (
        <PaintSizeOpacityControls
          opacity={1}
          setOpacity={vi.fn()}
          setSize={setSize}
          size={size}
          sizeLabel="Brush size"
        />
      );
    };
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <Harness />
        </ChakraProvider>
      );
    });

    const slider = page.getByRole('slider', { name: 'Brush size' });
    await act(async () => {
      await userEvent.click(slider);
      await userEvent.keyboard('{End}');
    });

    await expect.element(page.getByRole('spinbutton', { name: 'Brush size' })).toHaveValue('600');
    await expect.element(slider).toHaveAttribute('aria-valuetext', '600px');

    await act(async () => {
      await userEvent.keyboard('{ArrowRight}{PageUp}');
    });
    await expect.element(page.getByRole('spinbutton', { name: 'Brush size' })).toHaveValue('600');
    await expect.element(slider).toHaveAttribute('aria-valuetext', '600px');
  });

  it('supports PageUp and PageDown in logical brush units', async () => {
    const Harness = () => {
      const [size, setSize] = useState(0.1);
      return (
        <PaintSizeOpacityControls
          opacity={1}
          setOpacity={vi.fn()}
          setSize={setSize}
          size={size}
          sizeLabel="Brush size"
        />
      );
    };
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <Harness />
        </ChakraProvider>
      );
    });

    const slider = page.getByRole('slider', { name: 'Brush size' });
    await act(async () => {
      await userEvent.click(slider);
      await userEvent.keyboard('{PageUp}');
    });
    await expect.element(page.getByRole('spinbutton', { name: 'Brush size' })).toHaveValue('0.2');

    await act(async () => {
      await userEvent.keyboard('{PageDown}');
    });
    await expect.element(page.getByRole('spinbutton', { name: 'Brush size' })).toHaveValue('0.1');
  });

  it('does not turn an increase key into a decrease above the slider range', async () => {
    const Harness = () => {
      const [size, setSize] = useState(1000);
      return (
        <PaintSizeOpacityControls
          opacity={1}
          setOpacity={vi.fn()}
          setSize={setSize}
          size={size}
          sizeLabel="Brush size"
        />
      );
    };
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <Harness />
        </ChakraProvider>
      );
    });

    const slider = page.getByRole('slider', { name: 'Brush size' });
    await act(async () => {
      await userEvent.click(slider);
      await userEvent.keyboard('{ArrowRight}');
    });

    await expect.element(page.getByRole('spinbutton', { name: 'Brush size' })).toHaveValue('1000');
  });
});
