import type { CanvasDocumentContractV3, CanvasNodeContract } from '@workbench/canvas-engine/api';

import {
  groupContract,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { createInstance } from 'i18next';
import { describe, expect, it, vi } from 'vitest';

import {
  cropAllLayersToFrame,
  hasCroppableLayers,
  resolveCanvasContextMenu,
  resolveCanvasContextMenuBranch,
} from './canvasContextMenu';

const englishCatalogModules = import.meta.glob('../../../../public/locales/en.json', {
  eager: true,
  import: 'default',
});
const i18n = createInstance();
await i18n.init({
  initAsync: false,
  lng: 'en',
  resources: { en: { translation: Object.values(englishCatalogModules)[0] as Record<string, unknown> } },
});

const baseOptions = {
  clientX: 240,
  clientY: 160,
  isInlineEditor: false,
  isInteractionLocked: false,
};

describe('resolveCanvasContextMenu', () => {
  it('leaves inline editors to the native context menu', () => {
    expect(resolveCanvasContextMenu({ ...baseOptions, isInlineEditor: true, selectedLayerId: 'layer-1' })).toEqual({
      preventDefault: false,
      target: null,
    });
  });

  it('opens the global menu while interaction is locked, ignoring the selected layer', () => {
    expect(resolveCanvasContextMenu({ ...baseOptions, isInteractionLocked: true, selectedLayerId: 'layer-1' })).toEqual(
      {
        preventDefault: true,
        target: { layerId: null, x: 240, y: 160 },
      }
    );
  });

  it('opens the layer menu at the pointer for the SELECTED layer', () => {
    expect(resolveCanvasContextMenu({ ...baseOptions, selectedLayerId: 'layer-1' })).toEqual({
      preventDefault: true,
      target: { layerId: 'layer-1', x: 240, y: 160 },
    });
  });

  it('targets the selected layer regardless of where the pointer is', () => {
    const overOneCorner = resolveCanvasContextMenu({ ...baseOptions, selectedLayerId: 'layer-7' });
    const overAnother = resolveCanvasContextMenu({
      ...baseOptions,
      clientX: 900,
      clientY: 900,
      selectedLayerId: 'layer-7',
    });

    expect(overOneCorner.target?.layerId).toBe('layer-7');
    expect(overAnother.target?.layerId).toBe('layer-7');
  });

  it('opens the global menu when no layer is selected', () => {
    expect(resolveCanvasContextMenu({ ...baseOptions, selectedLayerId: null })).toEqual({
      preventDefault: true,
      target: { layerId: null, x: 240, y: 160 },
    });
  });

  it('opens the global menu without a target when the engine is unavailable', () => {
    const resolution = resolveCanvasContextMenu(baseOptions);

    expect(resolution).toEqual({
      preventDefault: true,
      target: { layerId: null, x: 240, y: 160 },
    });
    expect(resolveCanvasContextMenuBranch(resolution.target, false)).toBe('global');
  });
});

describe('hasCroppableLayers', () => {
  const documentWith = (nodes: CanvasNodeContract[]): CanvasDocumentContractV3 => ({
    ...createEmptyCanvasDocument(),
    stacks: stacksFrom(nodes),
  });

  it('finds a croppable leaf in any stack, hidden or disabled ones included', () => {
    expect(hasCroppableLayers(documentWith([layerContract('mask', 'inpaint_mask', { isEnabled: false })]))).toBe(true);
    expect(hasCroppableLayers(documentWith([groupContract('G', [layerContract('region', 'regional_guidance')])]))).toBe(
      true
    );
  });

  it('has nothing to crop when every leaf is locked by itself or by a group above it', () => {
    expect(hasCroppableLayers(documentWith([]))).toBe(false);
    expect(
      hasCroppableLayers(
        documentWith([
          layerContract('locked', 'raster', { isLocked: true }),
          groupContract('G', [layerContract('inside', 'control')], { isLocked: true }),
        ])
      )
    ).toBe(false);
  });
});

describe('cropAllLayersToFrame', () => {
  type CropResult = Awaited<ReturnType<Parameters<typeof cropAllLayersToFrame>[0]['cropAllLayersToBbox']>>;
  const notices = async (result: CropResult | Error) => {
    const notify = { error: vi.fn(), info: vi.fn(), success: vi.fn() };
    const cropAllLayersToBbox = () => (result instanceof Error ? Promise.reject(result) : Promise.resolve(result));
    await cropAllLayersToFrame({ cropAllLayersToBbox }, notify, i18n.t);
    return {
      error: notify.error.mock.calls,
      info: notify.info.mock.calls,
      success: notify.success.mock.calls,
    };
  };

  it.each([
    [{ cropped: 1, skippedLocked: 0 }, ['Cropped 1 layer to the frame', undefined]],
    [{ cropped: 4, skippedLocked: 1 }, ['Cropped 4 layers to the frame', '1 locked layer skipped']],
    [{ cropped: 2, skippedLocked: 3 }, ['Cropped 2 layers to the frame', '3 locked layers skipped']],
  ])('reports %o as cropped, with the locked layers it skipped', async (counts, expected) => {
    expect(await notices({ status: 'cropped', ...counts })).toEqual({ error: [], info: [], success: [expected] });
  });

  it('says nothing needed cropping, and that only unlocked layers were checked when some were skipped', async () => {
    expect((await notices({ skippedLocked: 0, status: 'nothing' })).info).toEqual([
      ['Every layer already fits the frame', undefined],
    ]);
    expect((await notices({ skippedLocked: 2, status: 'nothing' })).info).toEqual([
      ['Every unlocked layer already fits the frame', '2 locked layers skipped'],
    ]);
  });

  it('names the layer whose adjustments prevented the crop', async () => {
    const result = await notices({
      ownerId: 'haze',
      ownerName: 'Haze',
      reason: 'too-large',
      status: 'adjustment-refused',
    });
    expect(result.error).toHaveLength(1);
    expect(result.error[0]?.[1]).toContain('Haze');
    expect(result.info).toEqual([]);
    expect(result.success).toEqual([]);
  });

  it('explains refusals and a rejected crop with the shared layer notice', async () => {
    expect((await notices({ status: 'over-budget' })).error).toEqual([
      [i18n.t('widgets.layers.actions.actionFailed'), i18n.t('widgets.layers.actions.overBudget')],
    ]);
    expect((await notices(new Error('rejected'))).error).toEqual([
      [i18n.t('widgets.layers.actions.actionFailed'), i18n.t('widgets.layers.actions.operationFailed')],
    ]);
  });
});
