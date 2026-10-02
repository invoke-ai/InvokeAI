import { ChakraProvider, ScrollArea } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { wheelScrollsHorizontally } from './wheelScrollsHorizontally';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const render = async (contentWidth: number) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root!.render(
      <div ref={wheelScrollsHorizontally} data-testid="strip" style={{ overflowX: 'auto', width: 200 }}>
        <div style={{ height: 20, width: contentWidth }} />
      </div>
    )
  );
  return host.querySelector<HTMLDivElement>('[data-testid="strip"]')!;
};

const wheel = (element: HTMLElement, init: WheelEventInit): WheelEvent => {
  const event = new WheelEvent('wheel', { bubbles: true, cancelable: true, ...init });
  element.dispatchEvent(event);
  return event;
};

describe('wheelScrollsHorizontally', () => {
  it('turns a vertical wheel into sideways travel and claims the event', async () => {
    const strip = await render(1000);
    expect(wheel(strip, { deltaY: 40 }).defaultPrevented).toBe(true);
    expect(strip.scrollLeft).toBe(40);
    wheel(strip, { deltaMode: WheelEvent.DOM_DELTA_LINE, deltaY: -1 });
    expect(strip.scrollLeft).toBe(24);
  });

  it('leaves trackpad gestures with their own horizontal delta alone', async () => {
    const strip = await render(1000);
    expect(wheel(strip, { deltaX: 30, deltaY: 40 }).defaultPrevented).toBe(false);
    expect(strip.scrollLeft).toBe(0);
  });

  it('passes the wheel through when the content fits', async () => {
    const strip = await render(100);
    expect(wheel(strip, { deltaY: 40 }).defaultPrevented).toBe(false);
  });

  it('drives the scroll container of a Chakra scroll area through its viewport ref', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root!.render(
        <ChakraProvider value={system}>
          <ScrollArea.Root size="xs" w="200px">
            <ScrollArea.Viewport ref={wheelScrollsHorizontally} data-testid="viewport">
              <ScrollArea.Content>
                <div style={{ height: 20, width: 1000 }} />
              </ScrollArea.Content>
            </ScrollArea.Viewport>
            <ScrollArea.Scrollbar orientation="horizontal">
              <ScrollArea.Thumb />
            </ScrollArea.Scrollbar>
          </ScrollArea.Root>
        </ChakraProvider>
      )
    );
    const viewport = host.querySelector<HTMLElement>('[data-testid="viewport"]')!;
    expect(viewport.scrollWidth).toBeGreaterThan(viewport.clientWidth);
    wheel(viewport, { deltaY: 40 });
    expect(viewport.scrollLeft).toBe(40);
  });

  it('stops listening once the strip is gone', async () => {
    const strip = await render(1000);
    await act(() => root!.unmount());
    expect(wheel(strip, { deltaY: 40 }).defaultPrevented).toBe(false);
  });
});
