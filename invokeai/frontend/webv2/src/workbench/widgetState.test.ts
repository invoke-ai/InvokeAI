import { getGallerySettings } from '@features/gallery/contracts';
import { describe, expect, it } from 'vitest';

import type { WidgetInstanceContract } from './widgetContracts';

import {
  applyWidgetDefaults,
  getProjectWidgetValues,
  getResolvedProjectWidgetValues,
  resolveWidgetValues,
} from './widgetState';

const instance = (typeId: string, values: Record<string, unknown>): WidgetInstanceContract => ({
  createdAt: '2026-10-30T00:00:00.000Z',
  id: typeId,
  state: { id: typeId, label: typeId, values, version: 1 },
  typeId,
});

const project = (values: Record<string, unknown>) => ({ widgetInstances: { gallery: instance('gallery', values) } });

describe('resolveWidgetValues', () => {
  it('applies a user default to an instance without its own value, not to one that pinned it', () => {
    const defaults = { gallery: { thumbnailFit: 'aspect' } };
    const following = project({ showDateBoards: true });
    const pinned = project({ thumbnailFit: 'square' });

    expect(getGallerySettings(getResolvedProjectWidgetValues(following, 'gallery', defaults)).thumbnailFit).toBe(
      'aspect'
    );
    expect(getGallerySettings(getResolvedProjectWidgetValues(pinned, 'gallery', defaults)).thumbnailFit).toBe('square');
    // The pins themselves are untouched: persistence and write paths read these.
    expect(getProjectWidgetValues(following, 'gallery')).toEqual({ showDateBoards: true });
  });

  it("returns the instance's own values while its type has no defaults", () => {
    const gallery = instance('gallery', { thumbnailFit: 'square' });

    expect(resolveWidgetValues(gallery, {})).toBe(gallery.state.values);
    expect(resolveWidgetValues(gallery, { canvas: { showGrid: true } })).toBe(gallery.state.values);
    expect(resolveWidgetValues(gallery, { gallery: {} })).toBe(gallery.state.values);
  });

  it('returns the same merged object while the pins and defaults are unchanged', () => {
    const gallery = instance('gallery', { showDateBoards: true });
    const defaults = { gallery: { thumbnailFit: 'aspect' } };
    const first = resolveWidgetValues(gallery, defaults);

    expect(first).toEqual({ showDateBoards: true, thumbnailFit: 'aspect' });
    expect(resolveWidgetValues(gallery, defaults)).toBe(first);
    expect(resolveWidgetValues(gallery, { gallery: { thumbnailFit: 'aspect' } })).not.toBe(first);
    expect(resolveWidgetValues(instance('gallery', { showDateBoards: true }), defaults)).not.toBe(first);
  });

  it('reads a missing instance as the defaults alone, keeping identity across reads', () => {
    const defaults = { gallery: { thumbnailFit: 'aspect' } };
    const empty = { widgetInstances: {} };

    expect(resolveWidgetValues(undefined, defaults)).toEqual({});
    expect(getResolvedProjectWidgetValues(empty, 'gallery', {})).toEqual({});
    expect(getResolvedProjectWidgetValues(empty, 'gallery', defaults)).toEqual({ thumbnailFit: 'aspect' });
    expect(getResolvedProjectWidgetValues(empty, 'gallery', defaults)).toBe(
      getResolvedProjectWidgetValues({ widgetInstances: {} }, 'gallery', defaults)
    );
  });
});

describe('applyWidgetDefaults', () => {
  it('merges only the layerable keys of the widget type', () => {
    const values = { selectedBoardId: 'pinned-board' };
    const defaults = { gallery: { galleryPage: 7, selectedBoardId: 'stray', showDateBoards: true } };

    expect(applyWidgetDefaults(values, 'gallery', defaults)).toEqual({
      selectedBoardId: 'pinned-board',
      showDateBoards: true,
    });
    expect(applyWidgetDefaults({}, 'gallery', { gallery: { galleryPage: 7 } })).toEqual({});
  });

  it("never merges the project's generation defaults into the Generate widget's values", () => {
    const values = { model: 'm' };

    expect(applyWidgetDefaults(values, 'generate', { generate: { useNsfwChecker: true } })).toBe(values);
  });
});
