import { describe, expect, it } from 'vitest';

import {
  applyGalleryMarksPatch,
  chooseGalleryColorLabel,
  chooseGalleryFilterRating,
  chooseGalleryRating,
  cycleGalleryFilterRatingOperator,
  EMPTY_GALLERY_MARKS_FILTER,
  getGalleryGridListingMarks,
  getGalleryMarksFilter,
  getGalleryMarksFilterValues,
  getGalleryRevealMarksFilter,
  getGalleryStripListingMarks,
  matchesGalleryListingMarks,
  stepGalleryRating,
  toggleGalleryBookmark,
  toggleGalleryBookmarkedOnly,
  toggleGalleryReject,
  type GalleryMarks,
} from './marks';

const marks = (overrides: Partial<GalleryMarks> = {}): GalleryMarks => ({
  colorLabel: null,
  flag: 'none',
  rating: 0,
  ...overrides,
});

describe('mark semantics', () => {
  it('clears a flag only when every target has it, counting an unknown target as not having it', () => {
    expect(toggleGalleryBookmark([marks({ flag: 'pick' }), marks({ flag: 'pick' })])).toEqual({ flag: 'none' });
    expect(toggleGalleryBookmark([marks({ flag: 'pick' }), marks({ flag: 'reject' })])).toEqual({ flag: 'pick' });
    expect(toggleGalleryBookmark([marks({ flag: 'pick' }), {}])).toEqual({ flag: 'pick' });
    expect(toggleGalleryReject([marks({ flag: 'reject' })])).toEqual({ flag: 'none' });
    expect(toggleGalleryReject([marks({ flag: 'pick' })])).toEqual({ flag: 'reject' });
  });

  it('clears a rating or label chosen again by pointer, and sets it otherwise', () => {
    expect(chooseGalleryRating([marks({ rating: 3 }), marks({ rating: 3 })], 3)).toEqual({ rating: 0 });
    expect(chooseGalleryRating([marks({ rating: 3 }), marks({ rating: 2 })], 3)).toEqual({ rating: 3 });
    expect(chooseGalleryColorLabel([marks({ colorLabel: 'red' })], 'red')).toEqual({ colorLabel: null });
    expect(chooseGalleryColorLabel([marks({ colorLabel: 'red' }), {}], 'red')).toEqual({ colorLabel: 'red' });
  });

  it('steps from the primary rating, clamped to 0–5', () => {
    expect(stepGalleryRating(marks({ rating: 2 }), 1)).toEqual({ rating: 3 });
    expect(stepGalleryRating(marks({ rating: 5 }), 1)).toEqual({ rating: 5 });
    expect(stepGalleryRating(marks({ rating: 0 }), -1)).toEqual({ rating: 0 });
    expect(stepGalleryRating(undefined, 1)).toEqual({ rating: 1 });
  });

  it('keeps bookmark and reject exclusive and leaves unpatched marks and unchanged items alone', () => {
    const item = { colorLabel: 'blue' as const, rating: 4 as const, rejected: false, starred: true };

    expect(applyGalleryMarksPatch(item, { flag: 'reject' })).toEqual({ ...item, rejected: true, starred: false });
    expect(applyGalleryMarksPatch(item, { colorLabel: 'blue', rating: 4 })).toBe(item);
  });
});

describe('marks filter', () => {
  it('reads an older stored starredOnly as the bookmarked flag, and prefers a stored marks filter', () => {
    expect(getGalleryMarksFilter({ starredOnly: true }).flags).toEqual(['pick']);
    expect(getGalleryMarksFilter({})).toEqual(EMPTY_GALLERY_MARKS_FILTER);
    expect(
      getGalleryMarksFilter({ marksFilter: { flags: ['reject', 'bogus'], rating: 9 }, starredOnly: true })
    ).toEqual({ ...EMPTY_GALLERY_MARKS_FILTER, flags: ['reject'], rating: 5 });
  });

  it('writes only the marks filter, keeping fields it does not know, so a stale starredOnly no longer applies', () => {
    const values = { marksFilter: { futureField: 'kept' }, starredOnly: true };
    const written = getGalleryMarksFilterValues(values, toggleGalleryBookmarkedOnly(getGalleryMarksFilter(values)));

    expect(written).toEqual({ marksFilter: expect.objectContaining({ flags: ['pick'], futureField: 'kept' }) });
    expect(getGalleryMarksFilter({ ...values, ...written }).flags).toEqual(['pick']);
    expect(
      getGalleryMarksFilter({ ...values, ...getGalleryMarksFilterValues(values, EMPTY_GALLERY_MARKS_FILTER) })
    ).toEqual(EMPTY_GALLERY_MARKS_FILTER);
  });

  it('partitions bookmarked items into the strip without a flag filter, and lists only matches with one', () => {
    const rated = chooseGalleryFilterRating(EMPTY_GALLERY_MARKS_FILTER, 3);

    expect(getGalleryGridListingMarks(rated)).toEqual({ minRating: 3, starred: false });
    expect(getGalleryStripListingMarks(rated)).toEqual({ minRating: 3, starred: true });
    expect(getGalleryGridListingMarks({ ...rated, flags: ['pick', 'reject'] })).toEqual({
      flags: ['pick', 'reject'],
      minRating: 3,
    });
    expect(getGalleryStripListingMarks({ ...rated, flags: ['pick'] })).toBeNull();
    const exactly = cycleGalleryFilterRatingOperator(rated);

    expect(getGalleryGridListingMarks(exactly)).toMatchObject({ maxRating: 3, minRating: 3 });
    expect(getGalleryGridListingMarks(cycleGalleryFilterRatingOperator(exactly))).toEqual({
      maxRating: 3,
      starred: false,
    });
    expect(chooseGalleryFilterRating(rated, 3).rating).toBeNull();
    expect(chooseGalleryFilterRating(rated, 0).rating).toBeNull();
  });

  it('matches listings as the backend filters them, unlabelled items included by "none"', () => {
    const item = marks({ flag: 'reject', rating: 2 });

    expect(matchesGalleryListingMarks({ flags: ['reject'], maxRating: 2 }, item)).toBe(true);
    expect(matchesGalleryListingMarks({ colorLabels: ['none'] }, item)).toBe(true);
    expect(matchesGalleryListingMarks({ colorLabels: ['red'] }, item)).toBe(false);
    expect(matchesGalleryListingMarks({ starred: true }, item)).toBe(false);
  });

  it('reveals an item in a filter that lists it in the grid, keeping what it still passes', () => {
    const filter = { ...chooseGalleryFilterRating(EMPTY_GALLERY_MARKS_FILTER, 2), flags: ['reject' as const] };

    expect(getGalleryRevealMarksFilter(filter, marks({ flag: 'reject', rating: 4 }))).toBe(filter);
    expect(getGalleryRevealMarksFilter(filter, marks({ flag: 'pick', rating: 4 }))).toEqual({
      ...filter,
      flags: ['pick'],
    });
    expect(getGalleryRevealMarksFilter(filter, marks({ rating: 1 }))).toEqual(EMPTY_GALLERY_MARKS_FILTER);
  });
});
