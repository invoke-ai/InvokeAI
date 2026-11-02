/**
 * Cooldown persists dirty layers. A bucket preview still on the canvas when it starts (a held press, or a released
 * fill waiting on the worker) must be put back first, so the uploaded pixels are the recorded ones.
 */

import type { CanvasDocumentContractV3, CanvasStateContractV3 } from '@workbench/canvas-engine/contracts';
import type { FloodFillExecutor } from '@workbench/canvas-engine/render/fill/floodFillService';
import type { CanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import type { Project } from '@workbench/projectContracts';

import { stacksFrom, layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createInlineFloodFillExecutor } from '@workbench/canvas-engine/render/fill/floodFillService.testing';
import { applyCanvasProjectMutation, type CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { describe, expect, it } from 'vitest';

import { createCanvasEngine } from './engine';

const documentWithPaint = (): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  selectedLayerId: 'paint',
  stacks: stacksFrom([layerContract('paint')]),
  version: 3,
  width: 100,
});

const createMutationPort = (document: CanvasDocumentContractV3): CanvasProjectMutationPort => {
  let project = {
    canvas: {
      document,
      documentRevision: 0,
      snapshots: [],
      stagingArea: {
        areThumbnailsVisible: false,
        autoSwitchMode: 'off',
        isVisible: false,
        pendingImageIds: [],
        pendingImages: [],
        selectedImageIndex: 0,
      },
      version: 3,
    } as CanvasStateContractV3,
    id: 'p1',
  } as unknown as Project;
  const listeners = new Set<() => void>();
  return {
    commitEdit: () => undefined,
    dispatch: (mutation: CanvasProjectMutation) => {
      const before = project.canvas;
      project = applyCanvasProjectMutation(project, mutation);
      listeners.forEach((listener) => listener());
      return project.canvas !== before;
    },
    getCanvasState: () => project.canvas,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
};

/** The inline worker, held while paused, so a test can keep a released fill waiting. */
const createPausableExecutor = () => {
  const inner = createInlineFloodFillExecutor();
  let gate: Promise<void> = Promise.resolve();
  let resume: () => void = () => undefined;
  const executor: FloodFillExecutor = {
    dispose: () => inner.dispose(),
    release: (pressId) => inner.release(pressId),
    run: (job) => gate.then(() => inner.run(job)),
  };
  return {
    executor,
    pause: () => {
      gate = new Promise((resolve) => {
        resume = resolve;
      });
    },
    resume: () => resume(),
  };
};

const settle = (): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, 0);
  });

const pointer = (type: string, x: number, buttons: number): PointerEvent =>
  new PointerEvent(type, { bubbles: true, buttons, clientX: x, clientY: 20, pointerId: 1, pointerType: 'mouse' });

/** The RGBA at the center of an uploaded PNG, and its size. */
const centerOf = async (blob: Blob): Promise<{ pixel: number[]; width: number; height: number }> => {
  const bitmap = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(bitmap, 0, 0);
  const pixel = [...ctx.getImageData(bitmap.width >> 1, bitmap.height >> 1, 1, 1).data];
  return { height: bitmap.height, pixel, width: bitmap.width };
};

describe('paint bucket and engine cooldown in Chromium', () => {
  it.each(['held', 'settling'] as const)('persists the recorded fill, not a %s preview', async (phase) => {
    const worker = createPausableExecutor();
    const uploads: Blob[] = [];
    const upload = (blob: Blob) => {
      uploads.push(blob);
      return Promise.resolve({ height: 100, imageName: `upload-${uploads.length}`, width: 100 });
    };
    const { engine } = createCanvasEngine({
      createFloodFillExecutor: () => worker.executor,
      fonts: null,
      imageResolver: () => Promise.reject(new Error('no images')),
      mutationPort: createMutationPort(documentWithPaint()),
      projectId: 'p1',
      reportError: () => undefined,
      uploadImage: upload,
      uploadIntermediateImage: upload,
    });
    const screen = document.createElement('canvas');
    const overlay = document.createElement('canvas');
    screen.width = overlay.width = 100;
    screen.height = overlay.height = 100;
    // Synthetic pointers have no capture to take.
    overlay.setPointerCapture = () => undefined;
    overlay.releasePointerCapture = () => undefined;
    engine.surface.attach(screen, overlay);
    engine.tools.setTool('paintBucket');
    const brush = (color: string) =>
      engine.interaction.set('brushOptions', { ...engine.interaction.get('brushOptions'), color });

    // A recorded red fill of the empty layer's whole frame.
    brush('#ff0000');
    overlay.dispatchEvent(pointer('pointerdown', 20, 1));
    overlay.dispatchEvent(pointer('pointerup', 20, 0));
    await settle();
    expect(engine.history.getEntries().past).toEqual(['Bucket fill']);

    // A blue fill of the same region, previewed on the canvas and never recorded.
    brush('#0000ff');
    overlay.dispatchEvent(pointer('pointerdown', 20, 1));
    await settle();
    if (phase === 'settling') {
      worker.pause();
      overlay.dispatchEvent(pointer('pointermove', 60, 1));
      overlay.dispatchEvent(pointer('pointerup', 60, 0));
    }

    expect(await engine.lifecycle.beginCooldown()).toBe('cooled');
    worker.resume();
    await settle();

    expect(engine.history.getEntries().past).toEqual(['Bucket fill']);
    const persisted = await centerOf(uploads.at(-1)!);
    expect(persisted).toEqual({ height: 100, pixel: [255, 0, 0, 255], width: 100 });
    engine.lifecycle.dispose();
  });
});
