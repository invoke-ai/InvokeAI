/* eslint-disable react/refs */
import { WHEEL_ZOOM_STEP } from '@workbench/panZoom';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { usePreviewLoupe } from './usePreviewLoupe';

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

const Harness = ({ naturalWidth = 200, token = 'a' }: { naturalWidth?: number; token?: string }) => {
  const loupe = usePreviewLoupe({ enabled: true, naturalWidth });
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
const mountHarness = async (): Promise<{ content: HTMLElement; stage: HTMLElement }> => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await interact(() => root?.render(<Harness />));

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
