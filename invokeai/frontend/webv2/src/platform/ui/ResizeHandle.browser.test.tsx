/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop */
import { Box, ChakraProvider, Flex } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, useRef, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { isResizeDragActive, ResizeHandle, type ResizeHandleProps, subscribeResizeDrag } from './ResizeHandle';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

type HarnessProps = Partial<Omit<ResizeHandleProps, 'onCommit' | 'paneRef'>> & { onCommit?: (value: number) => void };

const Harness = ({ onCommit, pane = 'before', unit = 'px', value: initial = 300, ...props }: HarnessProps) => {
  const [value, setValue] = useState(initial);
  const paneRef = useRef<HTMLDivElement>(null);
  const paneBox = <Box ref={paneRef} data-testid="pane" flexShrink={0} w={`${value}${unit}`} />;
  const handle = (
    <ResizeHandle
      label="Resize pane"
      max={500}
      min={200}
      orientation="vertical"
      pane={pane}
      paneRef={paneRef}
      unit={unit}
      value={value}
      onCommit={(next) => {
        onCommit?.(next);
        setValue(next);
      }}
      {...props}
    />
  );

  return (
    <Flex h="200px" w="800px">
      {pane === 'before' ? paneBox : <Box flex="1" />}
      {handle}
      {pane === 'before' ? <Box flex="1" /> : paneBox}
    </Flex>
  );
};

const mount = async (props: HarnessProps = {}) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <Harness {...props} />
      </ChakraProvider>
    )
  );
  const separator = host.querySelector<HTMLElement>('[role="separator"]');
  const pane = host.querySelector<HTMLElement>('[data-testid="pane"]');
  if (!separator || !pane) {
    throw new Error('harness did not render');
  }
  return { pane, separator };
};

const nextFrame = () =>
  act(
    () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() => resolve());
      })
  );

const pointer = (type: string, clientX: number, buttons = 1) =>
  new PointerEvent(type, { bubbles: true, buttons, clientX, pointerId: 1 });

const press = (separator: HTMLElement) => act(() => separator.dispatchEvent(pointer('pointerdown', 0)));
const moveTo = async (clientX: number) => {
  await act(() => window.dispatchEvent(pointer('pointermove', clientX)));
  await nextFrame();
};
const release = (clientX: number, type = 'pointerup') => act(() => window.dispatchEvent(pointer(type, clientX, 0)));

describe('ResizeHandle', () => {
  it('describes the divider to assistive tech', async () => {
    const { separator } = await mount();

    expect(separator.getAttribute('aria-label')).toBe('Resize pane');
    expect(separator.getAttribute('aria-orientation')).toBe('vertical');
    expect(separator.getAttribute('aria-valuemin')).toBe('200');
    expect(separator.getAttribute('aria-valuemax')).toBe('500');
    expect(separator.getAttribute('aria-valuenow')).toBe('300');
  });

  it('renders a drag live on the pane and commits once on release', async () => {
    const onCommit = vi.fn();
    const { pane, separator } = await mount({ onCommit });

    await press(separator);
    await moveTo(40);

    expect(pane.style.width).toBe('340px');
    expect(document.documentElement.hasAttribute('data-pointer-drag')).toBe(true);
    expect(onCommit).not.toHaveBeenCalled();

    await release(40);
    expect(onCommit).toHaveBeenCalledExactlyOnceWith(340);
    expect(document.documentElement.hasAttribute('data-pointer-drag')).toBe(false);

    await nextFrame();
    expect(pane.style.width).toBe('');
    expect(pane.getBoundingClientRect().width).toBe(340);
  });

  it('writes the pane at most once per frame however many moves arrive', async () => {
    const { pane, separator } = await mount();
    const writes: MutationRecord[] = [];
    const observer = new MutationObserver((records) => writes.push(...records));
    observer.observe(pane, { attributeFilter: ['style'] });

    await press(separator);
    await act(() => {
      for (const clientX of [5, 10, 15, 20, 25]) {
        window.dispatchEvent(pointer('pointermove', clientX));
      }
    });
    await nextFrame();
    observer.disconnect();

    expect(writes).toHaveLength(1);
    expect(pane.style.width).toBe('325px');
  });

  it('clamps a drag to its bounds', async () => {
    const onCommit = vi.fn();
    const { separator } = await mount({ onCommit });

    await press(separator);
    await moveTo(900);
    await release(900);

    expect(onCommit).toHaveBeenCalledExactlyOnceWith(500);
  });

  it('grows a pane after the divider toward the start', async () => {
    const onCommit = vi.fn();
    const { separator } = await mount({ onCommit, pane: 'after' });

    await press(separator);
    await moveTo(40);
    await release(40);

    expect(onCommit).toHaveBeenCalledExactlyOnceWith(260);
  });

  it('arms a collapse past the collapse point, backs off with hysteresis, and collapses on release', async () => {
    const onCollapse = vi.fn();
    const onCommit = vi.fn();
    const { pane, separator } = await mount({ collapse: { at: 150, onCollapse, preview: 0 }, onCommit });

    await press(separator);
    await moveTo(-149);
    expect(separator.hasAttribute('data-collapse-armed')).toBe(false);
    expect(pane.style.width).toBe('200px');

    await moveTo(-150);
    expect(separator.hasAttribute('data-collapse-armed')).toBe(true);
    expect(pane.style.width).toBe('0px');

    await moveTo(-110);
    expect(separator.hasAttribute('data-collapse-armed')).toBe(true);
    await moveTo(-109);
    expect(separator.hasAttribute('data-collapse-armed')).toBe(false);

    await moveTo(-200);
    await release(-200);
    expect(onCollapse).toHaveBeenCalledExactlyOnceWith('pointer');
    expect(onCommit).not.toHaveBeenCalled();
  });

  it('keeps the size but never collapses when the gesture is interrupted', async () => {
    const onCollapse = vi.fn();
    const onCommit = vi.fn();
    const { separator } = await mount({ collapse: { at: 150, onCollapse, preview: 0 }, onCommit });

    await press(separator);
    await moveTo(-200);
    await release(-200, 'pointercancel');

    expect(onCollapse).not.toHaveBeenCalled();
    expect(onCommit).toHaveBeenCalledExactlyOnceWith(200);
  });

  it('abandons the drag on Escape', async () => {
    const onCommit = vi.fn();
    const { pane, separator } = await mount({ onCommit });

    await press(separator);
    await moveTo(60);
    await act(() => window.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Escape' })));
    await release(60);
    await nextFrame();

    expect(onCommit).not.toHaveBeenCalled();
    expect(pane.style.width).toBe('');
    expect(separator.getAttribute('aria-valuenow')).toBe('300');
  });

  it('ends a drag whose release another window swallowed', async () => {
    const onCommit = vi.fn();
    const { separator } = await mount({ onCommit });

    await press(separator);
    await moveTo(30);
    await act(() => window.dispatchEvent(pointer('pointermove', 30, 0)));

    expect(onCommit).toHaveBeenCalledExactlyOnceWith(330);
  });

  it('steps by keyboard along its axis and collapses from the floor', async () => {
    const onCollapse = vi.fn();
    const onCommit = vi.fn();
    const { separator } = await mount({ collapse: { at: 150, onCollapse, preview: 0 }, onCommit });
    const key = (init: KeyboardEventInit) =>
      act(() => separator.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, ...init })));

    await key({ key: 'ArrowRight' });
    await key({ key: 'ArrowRight', shiftKey: true });
    await key({ key: 'ArrowLeft' });
    await key({ key: 'End' });
    await key({ key: 'Home' });
    expect(onCommit.mock.calls.map(([value]) => value)).toEqual([316, 348, 332, 500, 200]);

    await key({ key: 'ArrowLeft' });
    expect(onCollapse).toHaveBeenCalledExactlyOnceWith('keyboard');
  });

  it('converts pointer travel into a percentage of the container', async () => {
    const onCommit = vi.fn();
    const { pane, separator } = await mount({ max: 75, min: 10, onCommit, unit: '%', value: 25 });

    await press(separator);
    await moveTo(80);
    expect(pane.style.width).toBe('35%');

    await release(80);
    expect(onCommit).toHaveBeenCalledExactlyOnceWith(35);
  });

  it('announces the drag so costly layout can wait for it to end', async () => {
    const listener = vi.fn();
    const unsubscribe = subscribeResizeDrag(listener);
    const { separator } = await mount();

    await press(separator);
    expect(isResizeDragActive()).toBe(true);

    await moveTo(20);
    await release(20);
    unsubscribe();

    expect(isResizeDragActive()).toBe(false);
    expect(listener).toHaveBeenCalledTimes(2);
  });

  it('releases the page even when the commit throws', async () => {
    const swallow = (event: ErrorEvent) => event.preventDefault();
    window.addEventListener('error', swallow);
    const { pane, separator } = await mount({
      onCommit: () => {
        throw new Error('store rejected the size');
      },
    });

    await press(separator);
    await moveTo(40);
    await release(40);
    window.removeEventListener('error', swallow);

    expect(pane.style.width).toBe('');
    expect(isResizeDragActive()).toBe(false);
    expect(document.documentElement.hasAttribute('data-pointer-drag')).toBe(false);
  });

  it('applies a move released before its frame, and nothing after the release', async () => {
    const onCommit = vi.fn();
    const { pane, separator } = await mount({ onCommit });

    await press(separator);
    await act(() => window.dispatchEvent(pointer('pointermove', 40)));
    await release(40);
    expect(onCommit).toHaveBeenCalledExactlyOnceWith(340);

    await nextFrame();
    await nextFrame();
    expect(pane.style.width).toBe('');
  });

  it('does not reapply a pending move after a throwing commit restores the pane', async () => {
    const swallow = (event: ErrorEvent) => event.preventDefault();
    window.addEventListener('error', swallow);
    const { pane, separator } = await mount({
      onCommit: () => {
        throw new Error('store rejected the size');
      },
    });

    await press(separator);
    await act(() => window.dispatchEvent(pointer('pointermove', 40)));
    await release(40);
    await nextFrame();
    window.removeEventListener('error', swallow);

    expect(pane.style.width).toBe('');
  });

  it('commits nothing and releases the page when it unmounts mid-drag', async () => {
    const onCommit = vi.fn();
    const { separator } = await mount({ onCommit });

    await press(separator);
    await moveTo(40);
    await act(() => root?.unmount());
    await release(40);

    expect(onCommit).not.toHaveBeenCalled();
    expect(isResizeDragActive()).toBe(false);
    expect(document.documentElement.hasAttribute('data-pointer-drag')).toBe(false);
  });
});
