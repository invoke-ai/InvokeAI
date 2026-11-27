import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { forwardWheelToStrip, wheelScrollsHorizontally } from './wheelScrollsHorizontally';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const render = async (contentWidth: number, dir: 'ltr' | 'rtl' = 'ltr') => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root!.render(
      <div ref={wheelScrollsHorizontally} data-testid="strip" dir={dir} style={{ overflowX: 'auto', width: 200 }}>
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

  it('scrolls a strip for a wheel forwarded from something laid over it, horizontal gestures included', async () => {
    const strip = await render(1000, 'rtl');
    const forward = (init: WheelEventInit) => {
      const event = new WheelEvent('wheel', { cancelable: true, ...init });
      forwardWheelToStrip(strip, event);
      return event;
    };

    // A vertical wheel maps as on the strip itself: toward the end, negative in right-to-left.
    expect(forward({ deltaY: 40 }).defaultPrevented).toBe(true);
    expect(strip.scrollLeft).toBe(-40);
    // A trackpad swipe moves by its own physical delta, which the browser cannot do for an element outside the strip.
    forward({ deltaX: 16, deltaY: 40 });
    expect(strip.scrollLeft).toBe(-24);
  });
});
