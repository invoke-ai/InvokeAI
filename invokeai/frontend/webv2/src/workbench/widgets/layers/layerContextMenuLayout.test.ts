import type {
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { describe, expect, it } from 'vitest';

import { getLayerContextActions } from './layerContextActions';
import {
  getLayerContextMenuLayerLabelKey,
  getLayerContextMenuGroupLayout,
  getLayerContextMenuLayout,
  getLayerContextMenuRenderEntries,
  type LayerContextMenuSection,
} from './layerContextMenuLayout';
import {
  createControlLayer,
  createEmptyPaintLayer,
  createInpaintMaskLayer,
  createRegionalGuidanceLayer,
} from './layerOps';

const paintLayer = (id: string, patch: Partial<CanvasRasterLayerContractV2> = {}): CanvasLayerContract => ({
  ...createEmptyPaintLayer(id, id),
  source: { bitmap: { height: 10, imageName: `${id}.png`, width: 10 }, type: 'paint' },
  ...patch,
});

const makeDocument = (layers: CanvasLayerContract[]): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 512, width: 512, x: 0, y: 0 },
  height: 512,
  stacks: stacksFrom(layers),
  selectedLayerId: layers[0]?.id ?? null,
  version: 3,
  width: 512,
});

const actionsFor = (layer: CanvasLayerContract, layers: readonly CanvasLayerContract[] = [layer]) =>
  getLayerContextActions({
    canRunWorkflow: true,
    document: makeDocument([...layers]),
    hiddenByAncestor: false,
    hasEngine: true,
    hasSupportedContent: true,
    hasWorkflowBindings: true,
    interactionLocked: false,
    selectedIds: [layer.id],
    canGroupSelection: true,
    canDeleteSelection: true,
    canMergeSelection: true,
    layer,
    modelBase: null,
  });

const layoutFor = (layer: CanvasLayerContract, layers: readonly CanvasLayerContract[] = [layer]) =>
  getLayerContextMenuLayout(actionsFor(layer, layers));

const summarizeLayout = (layout: readonly LayerContextMenuSection[]) =>
  layout.map((section) => ({
    id: section.id,
    items: section.items.map((item) =>
      item.kind === 'action' ? item.action.id : `${item.id}(${item.actions.map((action) => action.id).join(',')})`
    ),
    presentation: section.presentation,
  }));

const summarize = (layer: CanvasLayerContract, layers: readonly CanvasLayerContract[] = [layer]) =>
  summarizeLayout(layoutFor(layer, layers));

describe('getLayerContextMenuLayout', () => {
  it('organizes raster actions like the legacy menu and keeps delete last', () => {
    const upper = paintLayer('upper');
    const below = paintLayer('below');

    expect(summarize(upper, [upper, below])).toEqual([
      {
        id: 'quick',
        items: ['arrange(move-to-front,move-forward,move-backward,move-to-back)', 'duplicate', 'group'],
        presentation: 'row',
      },
      {
        id: 'primary',
        items: [
          'transform',
          'rename',
          'fit-to-bbox',
          'add-adjustment(add-brightness-contrast,add-exposure,add-levels,add-curves,add-hsl,add-hue,add-invert)',
          'add-regenerate-region',
          'filter',
          'select-object',
          'run-workflow',
        ],
        presentation: 'list',
      },
      {
        id: 'operations',
        items: [
          'merge-down',
          'boolean(intersect,cutout,cutaway,exclude)',
          'copy-to(copy-to-clipboard,copy-to-control,copy-to-inpaint-mask,copy-to-regional-guidance)',
          'convert-to(convert-to-control,convert-to-inpaint-mask,convert-to-regional-guidance)',
        ],
        presentation: 'list',
      },
      { id: 'output', items: ['crop-to-bbox', 'save-to-assets'], presentation: 'list' },
      {
        id: 'state',
        items: [
          'toggle-visibility',
          'toggle-lock',
          'color-label(color-label-red,color-label-orange,color-label-yellow,color-label-green,color-label-blue,color-label-violet,color-label-gray,color-label-none)',
        ],
        presentation: 'list',
      },
      { id: 'danger', items: ['delete'], presentation: 'list' },
    ]);
  });

  it('keeps the primary section lean now the panes own the per-type additions', () => {
    const inpaint = createInpaintMaskLayer('Mask', 'mask');
    const regional = createRegionalGuidanceLayer('Region', 0, 'region');

    expect(summarize(inpaint)[1]).toEqual({
      id: 'primary',
      items: ['rename', 'fit-to-bbox', 'add-noise', 'add-denoise-limit', 'extract-masked-area'],
      presentation: 'list',
    });
    expect(summarize(regional)[1]).toEqual({
      id: 'primary',
      items: ['rename', 'fit-to-bbox', 'regional-auto-negative', 'add-reference-image'],
      presentation: 'list',
    });
  });

  it('places control filter and transparency actions before conversion operations', () => {
    const control = {
      ...createControlLayer('Control', 'control'),
      source: { image: { height: 10, imageName: 'control-image', width: 10 }, type: 'image' as const },
    };

    expect(summarize(control)[1]).toEqual({
      id: 'primary',
      items: [
        'transform',
        'rename',
        'fit-to-bbox',
        'filter',
        'select-object',
        'run-workflow',
        'control-transparency-effect',
      ],
      presentation: 'list',
    });
    expect(summarize(control)[2]).toEqual({
      id: 'operations',
      items: [
        'merge-down',
        'copy-to(copy-to-clipboard,copy-to-raster,copy-to-inpaint-mask,copy-to-regional-guidance)',
        'convert-to(convert-to-raster)',
      ],
      presentation: 'list',
    });
  });

  it('derives the same hierarchy when the input registry order is shuffled', () => {
    const upper = paintLayer('upper-shuffled');
    const below = paintLayer('below-shuffled');
    const actions = actionsFor(upper, [upper, below]);
    const reversed = [...actions].reverse();

    expect(summarizeLayout(getLayerContextMenuLayout(reversed))).toEqual(
      summarizeLayout(getLayerContextMenuLayout(actions))
    );
  });

  it('inserts optional canvas items immediately before the terminal danger section', () => {
    const upper = paintLayer('upper-with-canvas-items');
    const below = paintLayer('below-with-canvas-items');
    const layout = layoutFor(upper, [upper, below]);

    const summarizeEntries = (includeBeforeDangerSlot: boolean) =>
      getLayerContextMenuRenderEntries(layout, includeBeforeDangerSlot).map((entry) =>
        entry.kind === 'slot' ? entry.id : entry.section.id
      );

    expect(summarizeEntries(true)).toEqual([
      'quick',
      'primary',
      'operations',
      'output',
      'state',
      'before-danger',
      'danger',
    ]);
    expect(summarizeEntries(false)).toEqual(['quick', 'primary', 'operations', 'output', 'state', 'danger']);
  });

  it('builds legacy layer and Canvas groups while keeping row menus in one labeled layer group', () => {
    const upper = paintLayer('upper-with-groups');
    const below = paintLayer('below-with-groups');
    const layout = layoutFor(upper, [upper, below]);
    const summarizeEntries = (entries: ReturnType<typeof getLayerContextMenuGroupLayout>['layerEntries']) =>
      entries.map((entry) => (entry.kind === 'slot' ? entry.id : entry.section.id));

    const canvasMenu = getLayerContextMenuGroupLayout(layout, true);
    expect(summarizeEntries(canvasMenu.layerEntries)).toEqual(['quick', 'primary', 'operations', 'output', 'state']);
    expect(canvasMenu.hasCanvasGroup).toBe(true);
    expect(summarizeEntries(canvasMenu.trailingEntries)).toEqual(['danger']);

    const rowMenu = getLayerContextMenuGroupLayout(layout, false);
    expect(summarizeEntries(rowMenu.layerEntries)).toEqual([
      'quick',
      'primary',
      'operations',
      'output',
      'state',
      'danger',
    ]);
    expect(rowMenu.hasCanvasGroup).toBe(false);
    expect(rowMenu.trailingEntries).toEqual([]);
  });

  it.each([
    ['raster', 'widgets.canvas.import.raster'],
    ['control', 'widgets.canvas.import.control'],
    ['inpaint_mask', 'widgets.canvas.import.inpaintMask'],
    ['regional_guidance', 'widgets.canvas.import.regionalGuidance'],
  ] as const)('uses the legacy singular label for %s layer menus', (type, expected) => {
    expect(getLayerContextMenuLayerLabelKey(type)).toBe(expected);
  });
});
