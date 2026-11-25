import { describe, expect, it } from 'vitest';

import {
  doesGalleryCullMarkLeaveListing,
  GALLERY_CULL_COMMANDS,
  getGalleryCullMode,
  getGalleryCullPatch,
  getGalleryCullStep,
  isGalleryCullPressInverted,
  shouldGalleryCullAdvance,
  type GalleryCullCommandId,
} from './cull';
import {
  EMPTY_GALLERY_MARKS_FILTER,
  type GalleryItemMarkFields,
  type GalleryMarks,
  type GalleryMarksPatch,
} from './marks';

const marks = (overrides: Partial<GalleryMarks> = {}): GalleryMarks => ({
  colorLabel: null,
  flag: 'none',
  rating: 0,
  ...overrides,
});
const actionOf = (id: GalleryCullCommandId) => GALLERY_CULL_COMMANDS.find((command) => command.id === id)!.action;
const unmarkedItem: GalleryItemMarkFields = { colorLabel: null, rating: 0, rejected: false, starred: false };

describe('cull mode values', () => {
  it('reads values from before Cull mode as off with auto-advance on', () => {
    expect(getGalleryCullMode({})).toEqual({ autoAdvance: true, enabled: false });
    expect(getGalleryCullMode({ cullAutoAdvance: 'yes', cullMode: 1 })).toEqual({ autoAdvance: true, enabled: false });
    expect(getGalleryCullMode({ cullAutoAdvance: false, cullMode: true })).toEqual({
      autoAdvance: false,
      enabled: true,
    });
  });
});

describe('cull keys', () => {
  it('sets ratings and flags whatever the target holds, toggles labels, and steps from the primary', () => {
    const rated = [marks({ colorLabel: 'red', flag: 'pick', rating: 3 })];
    const patch = (id: GalleryCullCommandId, targets = rated) => getGalleryCullPatch(actionOf(id), targets, targets[0]);

    expect(patch('cull.rate3')).toEqual({ rating: 3 });
    expect(patch('cull.clearRating')).toEqual({ rating: 0 });
    expect(patch('cull.bookmark')).toEqual({ flag: 'pick' });
    expect(patch('cull.clearFlag')).toEqual({ flag: 'none' });
    expect(patch('cull.labelRed')).toEqual({ colorLabel: null });
    expect(patch('cull.labelBlue')).toEqual({ colorLabel: 'blue' });
    expect(patch('cull.lowerRating')).toEqual({ rating: 2 });
    // The primary is the last item added to the selection, not the first.
    expect(getGalleryCullPatch(actionOf('cull.raiseRating'), [marks({ rating: 5 }), marks()], marks())).toEqual({
      rating: 1,
    });
  });

  it('advances one target unless Shift inverts the switch, and never on a rating step or a selection of several', () => {
    const advances = (id: GalleryCullCommandId, autoAdvance: boolean, press: unknown, targetCount = 1) =>
      shouldGalleryCullAdvance(actionOf(id), { autoAdvance, inverted: isGalleryCullPressInverted(press), targetCount });

    expect(advances('cull.rate2', true, { shiftKey: false })).toBe(true);
    expect(advances('cull.rate2', true, { shiftKey: true })).toBe(false);
    expect(advances('cull.reject', false, { shiftKey: true })).toBe(true);
    expect(advances('cull.labelGreen', false, undefined)).toBe(false);
    expect(advances('cull.raiseRating', true, { shiftKey: true })).toBe(false);
    expect(advances('cull.rate2', true, { shiftKey: false }, 2)).toBe(false);
  });
});

describe('cull auto-advance', () => {
  it('moves on from an item that stays, and stays on the last one', () => {
    expect(getGalleryCullStep({ advance: true, hasNext: true, hasPrevious: true, leaves: false })).toBe('next');
    expect(getGalleryCullStep({ advance: true, hasNext: false, hasPrevious: true, leaves: false })).toBeNull();
    expect(getGalleryCullStep({ advance: false, hasNext: true, hasPrevious: true, leaves: false })).toBeNull();
  });

  it('lands on the item that takes the place of one leaving the listing, or the one before the last', () => {
    expect(getGalleryCullStep({ advance: false, hasNext: true, hasPrevious: true, leaves: true })).toBe('next');
    expect(getGalleryCullStep({ advance: true, hasNext: false, hasPrevious: true, leaves: true })).toBe('previous');
    expect(getGalleryCullStep({ advance: true, hasNext: false, hasPrevious: false, leaves: true })).toBeNull();
  });

  it('lets an item leave only when it fits neither the strip nor the grid', () => {
    const unflagged = { ...EMPTY_GALLERY_MARKS_FILTER, flags: ['none' as const] };
    const threeAndUp = { ...EMPTY_GALLERY_MARKS_FILTER, rating: 3 as const };
    const leaves = (item: typeof unmarkedItem, patch: GalleryMarksPatch, filter = EMPTY_GALLERY_MARKS_FILTER) =>
      doesGalleryCullMarkLeaveListing(item, patch, filter);

    // Between the strip and the grid it changes section, not listing.
    expect(leaves(unmarkedItem, { flag: 'pick' })).toBe(false);
    expect(leaves({ ...unmarkedItem, starred: true }, { flag: 'none' })).toBe(false);
    expect(leaves({ ...unmarkedItem, rating: 4 }, { flag: 'pick' }, threeAndUp)).toBe(false);
    expect(leaves({ ...unmarkedItem, rating: 4 }, { rating: 2 }, threeAndUp)).toBe(true);
    expect(leaves(unmarkedItem, { rating: 4 }, unflagged)).toBe(false);
    expect(leaves(unmarkedItem, { flag: 'reject' }, unflagged)).toBe(true);
  });
});
