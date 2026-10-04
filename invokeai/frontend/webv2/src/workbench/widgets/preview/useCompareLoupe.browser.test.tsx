import { WHEEL_ZOOM_STEP } from '@workbench/panZoom';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { useCompareLoupe } from './useCompareLoupe';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const interact = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 20);
    });
  });

const Harness = () => {
  const loupe = useCompareLoupe({ enabled: true, naturalWidth: 200 });

  return ([0, 1] as const).map((index) => {
    const pane = loupe.getPane(index)!;

    return (
      <div
        key={index}
        ref={pane.frameRefCallback}
        data-testid={`frame-${index}`}
        style={{ height: 150, overflow: 'hidden', width: 200 }}
        {...pane.frameProps}
      >
        <img
          ref={pane.imageRefCallback}
          alt={`pane-${index}`}
          height="150"
          src="data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs="
          style={{ display: 'block', height: 150, width: 200 }}
          width="200"
        />
      </div>
    );
  });
};

afterEach(async () => {
  await interact(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

/** Pane 0 is 200×150; `paneX` is measured from its left edge, at its mid height. */
const mountHarness = async (): Promise<{
  images: HTMLImageElement[];
  pane: HTMLElement;
  touch: (
    type: 'pointerdown' | 'pointermove' | 'pointerup' | 'lostpointercapture',
    pointerId: number,
    paneX: number,
    options?: { isPrimary?: boolean; target?: HTMLElement }
  ) => void;
}> => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await interact(() => root?.render(<Harness />));

  const pane = host.querySelector<HTMLElement>('[data-testid="frame-0"]')!;
  const rect = pane.getBoundingClientRect();

  return {
    images: Array.from(host.querySelectorAll<HTMLImageElement>('img')),
    pane,
    touch: (type, pointerId, paneX, { isPrimary = false, target = pane } = {}) => {
      target.dispatchEvent(
        new PointerEvent(type, {
          bubbles: true,
          button: type === 'pointermove' ? -1 : 0,
          cancelable: true,
          clientX: rect.left + paneX,
          clientY: rect.top + 75,
          isPrimary,
          pointerId,
          pointerType: 'touch',
        })
      );
    },
  };
};

describe('useCompareLoupe', () => {
  it('keeps both panes stationary when wheel zoom is already at its maximum', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await interact(() => root?.render(<Harness />));

    const frame = host.querySelector<HTMLElement>('[data-testid="frame-0"]')!;
    const images = Array.from(host.querySelectorAll<HTMLImageElement>('img'));
    const frameRect = frame.getBoundingClientRect();
    const wheelAtCenter = (deltaY: number) =>
      frame.dispatchEvent(
        new WheelEvent('wheel', {
          bubbles: true,
          cancelable: true,
          clientX: frameRect.left + frame.clientWidth / 2,
          clientY: frameRect.top + frame.clientHeight / 2,
          deltaY,
        })
      );

    await interact(() => wheelAtCenter(-Math.log(8) / WHEEL_ZOOM_STEP));
    const transformsAtMaximum = images.map((image) => image.style.transform);
    expect(transformsAtMaximum).toEqual(['translate(-700px, -525px) scale(8)', 'translate(-700px, -525px) scale(8)']);

    await interact(() => wheelAtCenter(-100));

    expect(images.map((image) => image.style.transform)).toEqual(transformsAtMaximum);
  });

  it('pinches both panes together around the fingers on one of them', async () => {
    const { images, touch } = await mountHarness();

    await interact(() => {
      touch('pointerdown', 1, 50, { isPrimary: true });
      touch('pointerdown', 2, 150);
    });
    await interact(() => {
      touch('pointermove', 1, 0);
      touch('pointermove', 2, 200);
    });

    expect(images.map((image) => image.style.transform)).toEqual([
      'translate(-100px, -75px) scale(2)',
      'translate(-100px, -75px) scale(2)',
    ]);
  });

  it('keeps pinching when a pane loses the capture it handed to the other', async () => {
    const { images, touch } = await mountHarness();

    await interact(() => {
      touch('pointerdown', 1, 50, { isPrimary: true });
      touch('pointerdown', 2, 150);
    });
    // Capture moving between the panes — which is what arming a pinch whose
    // fingers are on different images does — fires `lostpointercapture` on the
    // pane that held it. Treating that as the finger going up would kill the
    // gesture before its first move.
    await interact(() => touch('lostpointercapture', 1, 50));
    await interact(() => {
      touch('pointermove', 1, 0);
      touch('pointermove', 2, 200);
    });

    expect(images.map((image) => image.style.transform)).toEqual([
      'translate(-100px, -75px) scale(2)',
      'translate(-100px, -75px) scale(2)',
    ]);
  });

  it('hands the pinch over to a one-finger pan, and ends it off the pane', async () => {
    const { images, touch } = await mountHarness();

    await interact(() => {
      touch('pointerdown', 1, 50, { isPrimary: true });
      touch('pointerdown', 2, 150);
    });
    await interact(() => {
      touch('pointermove', 1, -50);
      touch('pointermove', 2, 250);
    });
    expect(images[0]?.style.transform).toBe('translate(-200px, -150px) scale(3)');

    await interact(() => touch('pointerup', 2, 250));
    // Track the remaining finger through pan and release outside both panes.
    await interact(() => touch('pointermove', 1, -30));
    expect(images[0]?.style.transform).toBe('translate(-180px, -150px) scale(3)');

    await interact(() => touch('pointerup', 1, -30, { target: document.body }));
    await interact(() => touch('pointermove', 1, 100));

    expect(images[0]?.style.transform).toBe('translate(-180px, -150px) scale(3)');
  });
});
