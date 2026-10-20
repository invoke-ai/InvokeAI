/* eslint-disable react/refs */
import { WHEEL_ZOOM_SNAP_IDLE_MS, WHEEL_ZOOM_STEP } from '@workbench/panZoom';
import { act, createRef, type RefObject } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { usePreviewLoupe, type PreviewLoupeControls } from './usePreviewLoupe';

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

const Harness = ({
  controlsRef,
  naturalWidth = 200,
  token = 'a',
}: {
  controlsRef?: RefObject<PreviewLoupeControls | null>;
  naturalWidth?: number;
  token?: string;
}) => {
  const loupe = usePreviewLoupe({ controlsRef, enabled: true, naturalWidth });
  loupe.syncDisplayedSource(token);

  return (
    <div
      ref={loupe.stageRefCallback}
      data-testid="stage"
      style={{
        alignItems: 'center',
        display: 'flex',
        height: 300,
        justifyContent: 'center',
        overflow: 'hidden',
        position: 'relative',
        width: 400,
      }}
      {...loupe.stageProps}
    >
      <div ref={loupe.contentRefCallback} data-testid="content" style={{ height: 150, width: 200 }} />
    </div>
  );
};

afterEach(async () => {
  await interact(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

/**
 * The stage is 400×300 with a 200×150 content box centred in it, so the content
 * box's origin sits at (100, 75) in stage space. Coordinates below are given in
 * stage space and converted to client space against the live rect.
 */
const mountHarness = async ({
  controlsRef,
  naturalWidth,
}: { controlsRef?: RefObject<PreviewLoupeControls | null>; naturalWidth?: number } = {}): Promise<{
  content: HTMLElement;
  stage: HTMLElement;
}> => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await interact(() => root?.render(<Harness controlsRef={controlsRef} naturalWidth={naturalWidth} />));

  return {
    content: host.querySelector<HTMLElement>('[data-testid="content"]')!,
    stage: host.querySelector<HTMLElement>('[data-testid="stage"]')!,
  };
};

const touch = (
  stage: HTMLElement,
  type: 'pointerdown' | 'pointermove' | 'pointerup',
  pointerId: number,
  stageX: number,
  stageY: number,
  { isPrimary = false, target = stage }: { isPrimary?: boolean; target?: HTMLElement } = {}
): void => {
  const rect = stage.getBoundingClientRect();

  target.dispatchEvent(
    new PointerEvent(type, {
      bubbles: true,
      button: type === 'pointermove' ? -1 : 0,
      cancelable: true,
      clientX: rect.left + stageX,
      clientY: rect.top + stageY,
      isPrimary,
      pointerId,
      pointerType: 'touch',
    })
  );
};

/** Two fingers down, 100px apart, centred on the content box's centre. */
const startPinch = (stage: HTMLElement): void => {
  touch(stage, 'pointerdown', 1, 150, 150, { isPrimary: true });
  touch(stage, 'pointerdown', 2, 250, 150);
};

describe('usePreviewLoupe', () => {
  it('does not pan when wheel zoom is already at its maximum', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await interact(() => root?.render(<Harness />));

    const stage = host.querySelector<HTMLElement>('[data-testid="stage"]')!;
    const content = host.querySelector<HTMLElement>('[data-testid="content"]')!;
    const stageRect = stage.getBoundingClientRect();
    const wheelAtCenter = (deltaY: number) =>
      stage.dispatchEvent(
        new WheelEvent('wheel', {
          bubbles: true,
          cancelable: true,
          clientX: stageRect.left + stage.clientWidth / 2,
          clientY: stageRect.top + stage.clientHeight / 2,
          deltaY,
        })
      );

    await interact(() => wheelAtCenter(-Math.log(8) / WHEEL_ZOOM_STEP));
    const transformAtMaximum = content.style.transform;
    expect(transformAtMaximum).toContain('scale(8)');

    await interact(() => wheelAtCenter(-100));

    expect(content.style.transform).toBe(transformAtMaximum);
  });

  it("zooms by the fingers' spread around the point they pinched", async () => {
    const { content, stage } = await mountHarness();

    await interact(() => startPinch(stage));
    await interact(() => {
      touch(stage, 'pointermove', 1, 100, 150);
      touch(stage, 'pointermove', 2, 300, 150);
    });

    expect(content.style.transform).toBe('translate(-100px, -75px) scale(2)');
  });

  it('carries the image with the fingers while they pinch', async () => {
    const { content, stage } = await mountHarness();

    await interact(() => startPinch(stage));
    await interact(() => {
      touch(stage, 'pointermove', 1, 50, 150);
      touch(stage, 'pointermove', 2, 350, 150);
    });
    expect(content.style.transform).toBe('translate(-200px, -150px) scale(3)');

    await interact(() => {
      touch(stage, 'pointermove', 1, 90, 150);
      touch(stage, 'pointermove', 2, 390, 150);
    });

    expect(content.style.transform).toBe('translate(-160px, -150px) scale(3)');
  });

  it('hands a pinch over to a one-finger pan when the second finger lifts', async () => {
    const { content, stage } = await mountHarness();

    await interact(() => startPinch(stage));
    await interact(() => {
      touch(stage, 'pointermove', 1, 50, 150);
      touch(stage, 'pointermove', 2, 350, 150);
    });
    await interact(() => touch(stage, 'pointerup', 2, 350, 150));

    // Continue panning from the remaining finger's current position after pinch.
    await interact(() => touch(stage, 'pointermove', 1, 70, 160));

    expect(content.style.transform).toBe('translate(-180px, -140px) scale(3)');
  });

  it('does not pair a finger whose release was never seen with the next touch', async () => {
    const { content, stage } = await mountHarness();

    // Clear missed pointer endings on the next first touch to prevent phantom pinches.
    await interact(() => touch(stage, 'pointerdown', 1, 150, 150, { isPrimary: true }));
    await interact(() => touch(stage, 'pointerdown', 2, 250, 150, { isPrimary: true }));
    await interact(() => touch(stage, 'pointermove', 2, 350, 150));

    expect(content.style.transform).toBe('');
  });

  it('pinches from where a finger has moved to, not where it left the stage', async () => {
    const { content, stage } = await mountHarness();

    await interact(() => touch(stage, 'pointerdown', 1, 150, 150, { isPrimary: true }));
    // Track off-stage positions so pinch separation reflects actual finger distance.
    await interact(() => touch(stage, 'pointermove', 1, 50, 150, { target: document.body }));
    await interact(() => touch(stage, 'pointerdown', 2, 250, 150));
    await interact(() => {
      touch(stage, 'pointermove', 1, -50, 150, { target: document.body });
      touch(stage, 'pointermove', 2, 550, 150);
    });

    expect(content.style.transform).toContain('scale(3)');
  });

  it('ends a gesture whose finger lifts away from the stage', async () => {
    const { content, stage } = await mountHarness();

    await interact(() => startPinch(stage));
    await interact(() => {
      touch(stage, 'pointermove', 1, 50, 150);
      touch(stage, 'pointermove', 2, 350, 150);
    });
    await interact(() => touch(stage, 'pointerup', 2, 350, 150));
    // End panning on releases outside the stage to prevent unrelated later movement from dragging media.
    await interact(() => touch(stage, 'pointerup', 1, 50, 150, { target: document.body }));
    const transformAtRelease = content.style.transform;

    await interact(() => touch(stage, 'pointermove', 1, 200, 150));

    expect(transformAtRelease).toBe('translate(-200px, -150px) scale(3)');
    expect(content.style.transform).toBe(transformAtRelease);
  });
});

describe('usePreviewLoupe across image swaps', () => {
  it('clears the old zoom when the displayed image changes, even though the image size changed with it', async () => {
    const { content, stage } = await mountHarness();
    const rect = stage.getBoundingClientRect();

    await interact(() =>
      stage.dispatchEvent(
        new WheelEvent('wheel', {
          bubbles: true,
          cancelable: true,
          clientX: rect.left + 200,
          clientY: rect.top + 150,
          deltaY: -600,
        })
      )
    );
    expect(content.style.transform).not.toBe('');

    // A new source with a different natural width: the reset scheduled for it
    // must survive the gesture math changing identity.
    await interact(() => root?.render(<Harness naturalWidth={400} token="b" />));
    await interact(() => undefined);

    expect(content.style.transform).toBe('');
    expect(content.style.imageRendering).toBe('');
  });
});

const wheelAt = (stage: HTMLElement, deltaY: number, { ctrlKey = false }: { ctrlKey?: boolean } = {}): void => {
  const rect = stage.getBoundingClientRect();

  stage.dispatchEvent(
    new WheelEvent('wheel', {
      bubbles: true,
      cancelable: true,
      clientX: rect.left + stage.clientWidth / 2,
      clientY: rect.top + stage.clientHeight / 2,
      ctrlKey,
      deltaY,
    })
  );
};

/** The wheel delta that takes actual zoom from `from` to `to`, unsnapped. */
const deltaBetween = (from: number, to: number): number => -Math.log(to / from) / WHEEL_ZOOM_STEP;

describe('usePreviewLoupe below fit', () => {
  // A 100px-wide image fitted into the 200px content box shows at 200%.
  it('shows an image the fit enlarges at its own size, centred, when asked for 100%', async () => {
    const controlsRef = createRef<PreviewLoupeControls>();
    const { content } = await mountHarness({ controlsRef, naturalWidth: 100 });

    await interact(() => controlsRef.current?.zoomTo(1));

    // Half the fitted size, so 100×75 in the 400×300 stage: its origin at (150, 112.5), from the box's (100, 75).
    expect(content.style.transform).toBe('translate(50px, 37.5px) scale(0.5)');
    expect(content.getBoundingClientRect().width).toBeCloseTo(100, 6);
  });

  it('lets wheel zoom-out go below fit but stop at 100%', async () => {
    const { content, stage } = await mountHarness({ naturalWidth: 100 });

    await interact(() => wheelAt(stage, 2000));

    expect(content.style.transform).toBe('translate(50px, 37.5px) scale(0.5)');
  });

  it('still stops at fit when the fit shows the image below its own size', async () => {
    const { content, stage } = await mountHarness({ naturalWidth: 400 });

    await interact(() => wheelAt(stage, 2000));

    expect(content.style.transform).toBe('');
  });
});

describe('usePreviewLoupe wheel snapping', () => {
  // A 400px-wide image fitted into the 200px content box shows at 50%, so scale 2 is 100%.
  it('rests on 100% of the image across small wheel steps, then continues from the wheel', async () => {
    const { content, stage } = await mountHarness({ naturalWidth: 400 });

    await interact(() => wheelAt(stage, deltaBetween(0.5, 0.99)));
    expect(content.style.transform).toContain('scale(2)');

    await interact(() => wheelAt(stage, -5));
    expect(content.style.transform).toContain('scale(2)');

    // The intended zoom is now 99% × e^(0.0015 × 25) ≈ 102.8%, outside the 2% tolerance.
    await interact(() => wheelAt(stage, -20));
    const scale = Number(/scale\(([^)]+)\)/.exec(content.style.transform)?.[1]);
    expect(scale / 2).toBeCloseTo(0.99 * Math.exp(WHEEL_ZOOM_STEP * 25), 6);
  });

  it('starts a new wheel gesture from 100% after a pause, releasing the movement left over on it', async () => {
    const { content, stage } = await mountHarness({ naturalWidth: 400 });

    // Rests on 100% with the wheel asking for 101%.
    await interact(() => wheelAt(stage, deltaBetween(0.5, 1.01)));
    expect(content.style.transform).toContain('scale(2)');

    // Within the gesture, -8 would ask for 101% × e^0.012 ≈ 102.2% and leave 100%; after a pause it starts from the
    // 100% shown, asks for about 101.2%, and stays.
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, WHEEL_ZOOM_SNAP_IDLE_MS + 50);
    });
    await interact(() => wheelAt(stage, -8));
    expect(content.style.transform).toContain('scale(2)');
  });

  it('does not snap a trackpad pinch', async () => {
    const { content, stage } = await mountHarness({ naturalWidth: 400 });

    await interact(() => wheelAt(stage, deltaBetween(0.5, 0.99) / 4, { ctrlKey: true }));

    expect(content.style.transform).not.toContain('scale(2)');
  });
});
