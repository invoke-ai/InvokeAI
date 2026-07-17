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
});
