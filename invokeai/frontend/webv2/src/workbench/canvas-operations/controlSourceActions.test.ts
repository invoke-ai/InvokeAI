import { accountLifecycle } from '@platform/state/accountLifecycle';
import { afterEach, describe, expect, it } from 'vitest';

import type { uploadCanvasImage } from './backend/canvasImages';
import type { ControlLayerIntent } from './operationTypes';

import { fillControlFromBbox } from './controlSourceActions';
import { attachCanvasOperations } from './operationAccess';

const BBOX = { height: 512, width: 768, x: 31, y: 47 };

const engineWith = (composite: unknown) => {
  const intents: ControlLayerIntent[] = [];
  const engine = {
    document: { getDocument: () => ({ bbox: BBOX }) },
    exports: { exportRasterComposite: () => composite },
    lifecycle: { flushPendingUploads: () => Promise.resolve() },
    projectId: 'project-1',
  } as unknown as Parameters<typeof fillControlFromBbox>[0]['engine'];
  attachCanvasOperations(engine, {
    applyControlLayerIntent: (_layerId: string, intent: ControlLayerIntent) => {
      intents.push(intent);
      return 'updated';
    },
  } as never);
  return { engine, intents };
};

afterEach(() => {
  accountLifecycle.invalidate();
});

describe('fillControlFromBbox', () => {
  it('uploads the bbox composite as a control image and places it at the bbox origin', async () => {
    const { engine, intents } = engineWith({ blob: new Blob(['png']), rect: BBOX, status: 'ok' });
    const uploads: Parameters<typeof uploadCanvasImage>[1][] = [];
    const uploadImage: typeof uploadCanvasImage = (_blob, options) => {
      uploads.push(options);
      return Promise.resolve({ height: 512, imageName: 'bbox.png', width: 768 });
    };

    await expect(fillControlFromBbox({ engine, layerId: 'control-1', uploadImage })).resolves.toEqual({
      imageName: 'bbox.png',
      result: 'updated',
      status: 'applied',
    });
    expect(uploads[0]).toMatchObject({ imageCategory: 'control', projectId: 'project-1' });
    expect(intents).toEqual([
      {
        image: { height: 512, imageName: 'bbox.png', width: 768 },
        kind: 'replace-image',
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 31, y: 47 },
      },
    ]);
  });

  it('reports an empty bbox without uploading', async () => {
    const { engine, intents } = engineWith({ status: 'empty' });
    const uploadImage = () => Promise.reject(new Error('should not upload'));

    await expect(fillControlFromBbox({ engine, layerId: 'control-1', uploadImage })).resolves.toEqual({
      status: 'empty',
    });
    expect(intents).toEqual([]);
  });

  it('drops an upload that finishes after the account changed', async () => {
    accountLifecycle.activate('control-source-a');
    const { engine, intents } = engineWith({ blob: new Blob(['png']), rect: BBOX, status: 'ok' });
    const uploadImage: typeof uploadCanvasImage = () => {
      accountLifecycle.activate('control-source-b');
      return Promise.resolve({ height: 1, imageName: 'late.png', width: 1 });
    };

    await expect(fillControlFromBbox({ engine, layerId: 'control-1', uploadImage })).resolves.toEqual({
      status: 'stale',
    });
    expect(intents).toEqual([]);
  });
});
