/* eslint-disable react/react-compiler */
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

const Harness = () => {
  const loupe = usePreviewLoupe({ enabled: true, naturalWidth: 200 });

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
      <div ref={loupe.contentRef} data-testid="content" style={{ height: 150, width: 200 }} />
    </div>
  );
};

afterEach(async () => {
  await interact(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

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
});
