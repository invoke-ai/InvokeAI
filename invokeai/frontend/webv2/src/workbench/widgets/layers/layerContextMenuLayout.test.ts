import type { CanvasLayerContract, CanvasRasterLayerContractV2 } from '@workbench/types';

import { describe, expect, it } from 'vitest';

import { getLayerContextActions } from './layerContextActions';
import { getLayerContextMenuLayout } from './layerContextMenuLayout';
import {
  createControlLayer,
  createEmptyPaintLayer,
  createInpaintMaskLayer,
  createRegionalGuidanceLayer,
} from './layerOps';

const paintLayer = (id: string, patch: Partial<CanvasRasterLayerContractV2> = {}): CanvasLayerContract => ({
  ...createEmptyPaintLayer(id, id),
  ...patch,
});

const layoutFor = (layer: CanvasLayerContract, layers: readonly CanvasLayerContract[] = [layer]) => {
  const actions = getLayerContextActions({ hasEngine: true, index: layers.indexOf(layer), layer, layers });
  return getLayerContextMenuLayout(actions);
};

const summarize = (layer: CanvasLayerContract, layers: readonly CanvasLayerContract[] = [layer]) =>
  layoutFor(layer, layers).map((section) => ({
    id: section.id,
    items: section.items.map((item) =>
      item.kind === 'action' ? item.action.id : `${item.id}(${item.actions.map((action) => action.id).join(',')})`
    ),
    presentation: section.presentation,
  }));

describe('getLayerContextMenuLayout', () => {
  it('organizes raster actions like the legacy menu and keeps delete last', () => {
    const upper = paintLayer('upper');
    const below = paintLayer('below');

    expect(summarize(upper, [upper, below])).toEqual([
      {
        id: 'quick',
        items: ['arrange(move-to-front,move-forward,move-backward,move-to-back)', 'duplicate'],
        presentation: 'row',
      },
      { id: 'primary', items: ['transform', 'rename', 'fit-to-bbox'], presentation: 'list' },
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
      { id: 'state', items: ['toggle-visibility', 'toggle-lock'], presentation: 'list' },
      { id: 'danger', items: ['delete'], presentation: 'list' },
    ]);
  });

  it('groups inpaint modifiers and regional additions into type-specific submenus', () => {
    const inpaint = createInpaintMaskLayer('Mask', 'mask');
    const regional = createRegionalGuidanceLayer('Region', 0, 'region');

    expect(summarize(inpaint)[1]).toEqual({
      id: 'primary',
      items: [
        'add-modifiers(inpaint-noise,inpaint-denoise-limit)',
        'transform',
        'rename',
        'fit-to-bbox',
        'extract-masked-area',
      ],
      presentation: 'list',
    });
    expect(summarize(regional)[1]).toEqual({
      id: 'primary',
      items: [
        'add-regional(regional-positive-prompt,regional-negative-prompt,regional-reference-image)',
        'transform',
        'rename',
        'fit-to-bbox',
        'regional-auto-negative',
      ],
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
      items: ['transform', 'rename', 'fit-to-bbox', 'filter', 'control-transparency-effect'],
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
});
