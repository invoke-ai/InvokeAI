import { getLayoutPreset } from '@workbench/layoutPresets';
import { describe, expect, it } from 'vitest';

import { registryAvailability, withUnavailable } from './layoutFixtures.testing';
import { resolveLayout } from './resolveLayout';
import { normalizeStructure } from './structure';
import { getUnmetNeed } from './unmetNeeds';
import { frontRegion, setRegionCollapsed, showGuest } from './view';

const edit = normalizeStructure(getLayoutPreset('edit').snapshot);
const compose = normalizeStructure(getLayoutPreset('compose').snapshot);
const NEEDS_CANVAS = ['canvas'];
const geometry = { heightPx: 300, widthPx: 400, x: 20, y: 20 };

describe('getUnmetNeed', () => {
  it('reports a need the screen does not show, only while the widget itself is on screen', () => {
    const onScreen = resolveLayout(edit, undefined, registryAvailability);
    const away = frontRegion(undefined, edit, 'center', 'preview', registryAvailability);

    expect(getUnmetNeed(onScreen, 'layers', NEEDS_CANVAS, registryAvailability.getPresentation)).toBeNull();
    expect(
      getUnmetNeed(
        resolveLayout(edit, away, registryAvailability),
        'layers',
        NEEDS_CANVAS,
        registryAvailability.getPresentation
      )
    ).toEqual({ isPlacedHere: true, typeId: 'canvas' });
    expect(
      getUnmetNeed(
        resolveLayout(edit, setRegionCollapsed(away, edit, 'right', true), registryAvailability),
        'layers',
        NEEDS_CANVAS,
        registryAvailability.getPresentation
      )
    ).toBeNull();
  });

  it('says when the need is in no place here, and says nothing for a need this session cannot show', () => {
    const layersGuest = showGuest(undefined, compose, 'layers', { surface: 'window', typeId: 'layers' }, geometry);
    const layout = resolveLayout(compose, layersGuest, registryAvailability);

    expect(getUnmetNeed(layout, 'layers', NEEDS_CANVAS, registryAvailability.getPresentation)).toEqual({
      isPlacedHere: false,
      typeId: 'canvas',
    });
    expect(
      getUnmetNeed(layout, 'layers', NEEDS_CANVAS, withUnavailable(registryAvailability, ['canvas']).getPresentation)
    ).toBeNull();
  });
});
