import type { CanvasLayerContract } from '@workbench/canvas-engine/api';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createCanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import { createCanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { LayerThumbnail } from './LayerThumbnail';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const cleanups: (() => void | Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) {
    await cleanup();
  }
});

const SIZE = 200;

/** A real engine over a new project (its inpaint mask selected), attached to on-page canvases. */
const setup = async () => {
  const store = createWorkbenchStore();
  const projectId = store.getState().activeProjectId;
  const engine = createCanvasEngine({
    ensureProjectOnServer: () => Promise.resolve(),
    imageResolver: () => Promise.resolve(new Blob()),
    mutationPort: createCanvasProjectMutationPort(store, projectId),
    projectId,
    reportError: () => undefined,
  });
  cleanups.push(() => engine.lifecycle.dispose());
  const surface = document.createElement('div');
  surface.style.cssText = `position:fixed;left:0;top:0;width:${SIZE}px;height:${SIZE}px`;
  const [screen, overlay] = [document.createElement('canvas'), document.createElement('canvas')];
  for (const canvas of [screen, overlay]) {
    canvas.width = SIZE;
    canvas.height = SIZE;
    canvas.style.cssText = `position:absolute;inset:0;width:${SIZE}px;height:${SIZE}px`;
    surface.append(canvas);
  }
  document.body.append(surface);
  cleanups.push(() => surface.remove());
  engine.surface.attach(screen, overlay);
  engine.interaction.set('clipToBbox', false);

  const document_ = store.getState().projects[0]!.canvas.document;
  const mask = document_.stacks.inpaint_mask[0] as CanvasLayerContract;
  expect(document_.selectedLayerId).toBe(mask.id);

  const host = document.createElement('div');
  host.style.cssText = 'position:fixed;right:0;top:0;width:48px;height:48px';
  document.body.append(host);
  const root = createRoot(host);
  cleanups.push(async () => {
    await act(() => root.unmount());
    host.remove();
  });
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <LayerThumbnail engine={engine} layer={mask} />
      </ChakraProvider>
    )
  );

  const gesture = async (points: readonly [number, number][]) => {
    const fire = (type: string, [x, y]: readonly [number, number], buttons: number) =>
      overlay.dispatchEvent(
        new PointerEvent(type, { bubbles: true, buttons, clientX: x, clientY: y, isPrimary: true, pointerId: 1 })
      );
    await act(() => {
      fire('pointerdown', points[0]!, 1);
      for (const point of points.slice(1)) {
        fire('pointermove', point, 1);
      }
      fire('pointerup', points.at(-1)!, 0);
    });
  };
  /** Pixels the thumbnail shows in the mask's red fill (the checkerboard behind it is grey); 0 without a canvas. */
  const thumbnailCoverage = (): number => {
    const canvas = host.querySelector('canvas');
    if (!canvas || canvas.style.display === 'none' || canvas.width === 0) {
      return 0;
    }
    const { data } = canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height);
    let masked = 0;
    for (let index = 0; index < data.length; index += 4) {
      masked += data[index]! - data[index + 1]! > 40 ? 1 : 0;
    }
    return masked;
  };
  return { engine, gesture, thumbnailCoverage };
};

describe('LayerThumbnail', () => {
  it('redraws the mask thumbnail after a shape, a brush stroke, and their undo and redo', async () => {
    const { engine, gesture, thumbnailCoverage } = await setup();
    expect(thumbnailCoverage()).toBe(0);

    engine.tools.setTool('shape');
    await gesture([
      [40, 40],
      [70, 60],
      [100, 80],
    ]);
    await vi.waitFor(() => expect(thumbnailCoverage()).toBeGreaterThan(0));
    const afterShape = thumbnailCoverage();

    engine.tools.setTool('brush');
    await gesture([
      [150, 40],
      [150, 100],
      [150, 160],
    ]);
    await vi.waitFor(() => expect(thumbnailCoverage()).not.toBe(afterShape));
    const afterStroke = thumbnailCoverage();

    await act(async () => {
      await engine.history.undo();
      await engine.history.undo();
    });
    await vi.waitFor(() => expect(thumbnailCoverage()).toBe(0));

    await act(async () => {
      await engine.history.redo();
      await engine.history.redo();
    });
    await vi.waitFor(() => expect(thumbnailCoverage()).toBe(afterStroke));
  });
});
