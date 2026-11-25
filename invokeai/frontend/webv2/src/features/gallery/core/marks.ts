/** Mirrors the backend's `ColorLabel`, in display order. */
export const GALLERY_COLOR_LABELS = ['red', 'yellow', 'green', 'blue', 'purple'] as const;

export type GalleryColorLabel = (typeof GALLERY_COLOR_LABELS)[number];

/** `pick` is the bookmark, stored as `starred`; never together with `reject`. */
export const GALLERY_MARK_FLAGS = ['pick', 'none', 'reject'] as const;

export type GalleryMarkFlag = (typeof GALLERY_MARK_FLAGS)[number];

export type GalleryRating = 0 | 1 | 2 | 3 | 4 | 5;

export const GALLERY_MAX_RATING = 5;

export const GALLERY_RATINGS = [1, 2, 3, 4, 5] as const;

export interface GalleryMarks {
  colorLabel: GalleryColorLabel | null;
  flag: GalleryMarkFlag;
  rating: GalleryRating;
}

/** A field left out is not known: an item no loaded page holds. */
export type GalleryKnownMarks = Partial<GalleryMarks>;

/** A field left out stays as it is on each item. */
export type GalleryMarksPatch = Partial<GalleryMarks>;

export interface GalleryItemMarkFields {
  colorLabel: GalleryColorLabel | null;
  rating: GalleryRating;
  rejected: boolean;
  starred: boolean;
}

export const UNMARKED_GALLERY_ITEM_FIELDS: Readonly<Omit<GalleryItemMarkFields, 'starred'>> = {
  colorLabel: null,
  rating: 0,
  rejected: false,
};

const isColorLabel = (value: unknown): value is GalleryColorLabel =>
  GALLERY_COLOR_LABELS.includes(value as GalleryColorLabel);

export const toGalleryColorLabel = (value: unknown): GalleryColorLabel | null => (isColorLabel(value) ? value : null);

/** Anything else, an older record's missing field included, reads as unrated. */
export const toGalleryRating = (value: unknown): GalleryRating =>
  typeof value === 'number' && Number.isFinite(value)
    ? (Math.min(GALLERY_MAX_RATING, Math.max(0, Math.round(value))) as GalleryRating)
    : 0;

export const getGalleryItemMarks = (item: GalleryItemMarkFields): GalleryMarks => ({
  colorLabel: item.colorLabel,
  flag: item.starred ? 'pick' : item.rejected ? 'reject' : 'none',
  rating: item.rating,
});

/** The item fields a patch sets; a flag sets both `starred` and `rejected`. */
export const getGalleryItemMarkFieldsPatch = (patch: GalleryMarksPatch): Partial<GalleryItemMarkFields> => ({
  ...(patch.flag === undefined ? {} : { rejected: patch.flag === 'reject', starred: patch.flag === 'pick' }),
  ...(patch.rating === undefined ? {} : { rating: patch.rating }),
  ...('colorLabel' in patch && patch.colorLabel !== undefined ? { colorLabel: patch.colorLabel } : {}),
});

/** The same object when the patch changes nothing, so cache patches keep unchanged items' identity. */
export const applyGalleryMarksPatch = <Item extends GalleryItemMarkFields>(
  item: Item,
  patch: GalleryMarksPatch
): Item => {
  const fields = getGalleryItemMarkFieldsPatch(patch);
  const changes = (Object.keys(fields) as (keyof GalleryItemMarkFields)[]).some((key) => item[key] !== fields[key]);

  return changes ? { ...item, ...fields } : item;
};

const isSharedBy = <Key extends keyof GalleryMarks>(
  targets: readonly GalleryKnownMarks[],
  key: Key,
  value: GalleryMarks[Key]
): boolean => targets.length > 0 && targets.every((target) => target[key] === value);

/** What every target shares; a field the targets disagree on, or that one does not know, is left out. */
export const getSharedGalleryMarks = (targets: readonly GalleryKnownMarks[]): GalleryKnownMarks => {
  const [first] = targets;
  const shared: GalleryKnownMarks = {};

  if (!first) {
    return shared;
  }
  if (first.flag !== undefined && isSharedBy(targets, 'flag', first.flag)) {
    shared.flag = first.flag;
  }
  if (first.rating !== undefined && isSharedBy(targets, 'rating', first.rating)) {
    shared.rating = first.rating;
  }
  if (first.colorLabel !== undefined && isSharedBy(targets, 'colorLabel', first.colorLabel)) {
    shared.colorLabel = first.colorLabel;
  }

  return shared;
};

export const toggleGalleryBookmark = (targets: readonly GalleryKnownMarks[]): GalleryMarksPatch => ({
  flag: isSharedBy(targets, 'flag', 'pick') ? 'none' : 'pick',
});

export const toggleGalleryReject = (targets: readonly GalleryKnownMarks[]): GalleryMarksPatch => ({
  flag: isSharedBy(targets, 'flag', 'reject') ? 'none' : 'reject',
});

/** Keys: `p`, `x` and `u` set a flag whatever the targets hold. */
export const setGalleryFlag = (flag: GalleryMarkFlag): GalleryMarksPatch => ({ flag });

/** From a pointer: choosing the rating every target already has clears it. */
export const chooseGalleryRating = (
  targets: readonly GalleryKnownMarks[],
  rating: GalleryRating
): GalleryMarksPatch => ({
  rating: rating !== 0 && isSharedBy(targets, 'rating', rating) ? 0 : rating,
});

/** Keys `1`–`5` set, `0` clears. */
export const setGalleryRating = (rating: GalleryRating): GalleryMarksPatch => ({ rating });

/** `]` and `[`: one step from the primary item's rating, clamped, for every target. An unknown rating counts as 0. */
export const stepGalleryRating = (primary: GalleryKnownMarks | undefined, step: -1 | 1): GalleryMarksPatch => ({
  rating: toGalleryRating((primary?.rating ?? 0) + step),
});

/** Choosing the label every target already has clears it. */
export const chooseGalleryColorLabel = (
  targets: readonly GalleryKnownMarks[],
  colorLabel: GalleryColorLabel
): GalleryMarksPatch => ({
  colorLabel: isSharedBy(targets, 'colorLabel', colorLabel) ? null : colorLabel,
});

/** Marks a listing keeps, AND-combined as the backend's listing filters are; a field left out keeps everything. */
export interface GalleryListingMarks {
  colorLabels?: (GalleryColorLabel | 'none')[];
  flags?: GalleryMarkFlag[];
  maxRating?: GalleryRating;
  minRating?: GalleryRating;
  /** The grid/strip partition: the strip lists bookmarked items, the grid the rest. */
  starred?: boolean;
}

// Literals, not spreads: a module-scope spread is a side effect that keeps the barrels re-exporting this module.
const GALLERY_COLOR_LABEL_FILTERS = ['red', 'yellow', 'green', 'blue', 'purple', 'none'] as const satisfies readonly [
  ...typeof GALLERY_COLOR_LABELS,
  'none',
];

export type GalleryColorLabelFilter = (typeof GALLERY_COLOR_LABEL_FILTERS)[number];

/** Present only when set, in a fixed order, so equal filters share a query key and unfiltered keys stay as they were. */
export const canonicalizeGalleryListingMarks = (marks: GalleryListingMarks): GalleryListingMarks => {
  const flags = GALLERY_MARK_FLAGS.filter((flag) => marks.flags?.includes(flag));
  const colorLabels = GALLERY_COLOR_LABEL_FILTERS.filter((label) => marks.colorLabels?.includes(label));

  return {
    ...(colorLabels.length > 0 ? { colorLabels } : {}),
    ...(flags.length > 0 ? { flags } : {}),
    ...(marks.maxRating !== undefined ? { maxRating: marks.maxRating } : {}),
    ...(marks.minRating !== undefined ? { minRating: marks.minRating } : {}),
    ...(marks.starred !== undefined ? { starred: marks.starred } : {}),
  };
};

/** Mirrors the backend's listing filters. */
export const matchesGalleryListingMarks = (listing: GalleryListingMarks, marks: GalleryMarks): boolean =>
  (listing.starred === undefined || listing.starred === (marks.flag === 'pick')) &&
  (listing.flags === undefined || listing.flags.length === 0 || listing.flags.includes(marks.flag)) &&
  (listing.minRating === undefined || marks.rating >= listing.minRating) &&
  (listing.maxRating === undefined || marks.rating <= listing.maxRating) &&
  (listing.colorLabels === undefined ||
    listing.colorLabels.length === 0 ||
    listing.colorLabels.includes(marks.colorLabel ?? 'none'));

export type GalleryRatingOperator = 'atLeast' | 'exactly' | 'atMost';

const GALLERY_RATING_OPERATORS: readonly GalleryRatingOperator[] = ['atLeast', 'exactly', 'atMost'];

export interface GalleryMarksFilter {
  colorLabels: GalleryColorLabelFilter[];
  flags: GalleryMarkFlag[];
  /** Null keeps every rating. */
  rating: Exclude<GalleryRating, 0> | null;
  ratingOperator: GalleryRatingOperator;
}

export const EMPTY_GALLERY_MARKS_FILTER: GalleryMarksFilter = {
  colorLabels: [],
  flags: [],
  rating: null,
  ratingOperator: 'atLeast',
};

const PICK_ONLY_FILTER: GalleryMarksFilter = {
  colorLabels: [],
  flags: ['pick'],
  rating: null,
  ratingOperator: 'atLeast',
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const parseGalleryMarksFilter = (value: Record<string, unknown>): GalleryMarksFilter => {
  const rating = toGalleryRating(value.rating);

  return {
    colorLabels: GALLERY_COLOR_LABEL_FILTERS.filter(
      (label) => Array.isArray(value.colorLabels) && value.colorLabels.includes(label)
    ),
    flags: GALLERY_MARK_FLAGS.filter((flag) => Array.isArray(value.flags) && value.flags.includes(flag)),
    rating: rating === 0 ? null : rating,
    ratingOperator: GALLERY_RATING_OPERATORS.includes(value.ratingOperator as GalleryRatingOperator)
      ? (value.ratingOperator as GalleryRatingOperator)
      : 'atLeast',
  };
};

/** Values from before `marksFilter` have only `starredOnly`, whose `true` reads as the bookmarked flag. */
export const getGalleryMarksFilter = (values: Readonly<Record<string, unknown>>): GalleryMarksFilter =>
  isRecord(values.marksFilter)
    ? parseGalleryMarksFilter(values.marksFilter)
    : values.starredOnly === true
      ? PICK_ONLY_FILTER
      : EMPTY_GALLERY_MARKS_FILTER;

/**
 * The values a filter write sets; fields this build does not know stay. A stored `starredOnly` is left alone: once
 * `marksFilter` exists it is never read.
 */
export const getGalleryMarksFilterValues = (
  values: Readonly<Record<string, unknown>>,
  filter: GalleryMarksFilter
): { marksFilter: Record<string, unknown> } => ({
  marksFilter: {
    ...(isRecord(values.marksFilter) ? values.marksFilter : {}),
    colorLabels: [...filter.colorLabels],
    flags: [...filter.flags],
    rating: filter.rating,
    ratingOperator: filter.ratingOperator,
  },
});

export const isGalleryMarksFilterActive = (filter: GalleryMarksFilter): boolean =>
  filter.flags.length > 0 || filter.rating !== null || filter.colorLabels.length > 0;

const toggleEntry = <Value>(entries: readonly Value[], value: Value, order: readonly Value[]): Value[] =>
  order.filter((entry) => (entry === value ? !entries.includes(value) : entries.includes(entry)));

export const toggleGalleryFilterFlag = (filter: GalleryMarksFilter, flag: GalleryMarkFlag): GalleryMarksFilter => ({
  ...filter,
  flags: toggleEntry(filter.flags, flag, GALLERY_MARK_FLAGS),
});

export const toggleGalleryFilterColorLabel = (
  filter: GalleryMarksFilter,
  colorLabel: GalleryColorLabelFilter
): GalleryMarksFilter => ({
  ...filter,
  colorLabels: toggleEntry(filter.colorLabels, colorLabel, GALLERY_COLOR_LABEL_FILTERS),
});

/** Choosing the active rating, or 0, clears it. */
export const chooseGalleryFilterRating = (filter: GalleryMarksFilter, rating: GalleryRating): GalleryMarksFilter => ({
  ...filter,
  rating: rating === 0 || filter.rating === rating ? null : rating,
});

export const cycleGalleryFilterRatingOperator = (filter: GalleryMarksFilter): GalleryMarksFilter => ({
  ...filter,
  ratingOperator:
    GALLERY_RATING_OPERATORS[(GALLERY_RATING_OPERATORS.indexOf(filter.ratingOperator) + 1) % 3] ?? 'atLeast',
});

/** `gallery.toggleStarredOnly`: the flags between bookmarked and none; rating and labels stay. */
export const toggleGalleryBookmarkedOnly = (filter: GalleryMarksFilter): GalleryMarksFilter => ({
  ...filter,
  flags: filter.flags.length === 1 && filter.flags[0] === 'pick' ? [] : ['pick'],
});

const getRatingBounds = ({ rating, ratingOperator }: GalleryMarksFilter): GalleryListingMarks =>
  rating === null
    ? {}
    : ratingOperator === 'exactly'
      ? { maxRating: rating, minRating: rating }
      : ratingOperator === 'atMost'
        ? { maxRating: rating }
        : { minRating: rating };

const getRatingAndLabelMarks = (filter: GalleryMarksFilter): GalleryListingMarks => ({
  ...getRatingBounds(filter),
  ...(filter.colorLabels.length > 0 ? { colorLabels: [...filter.colorLabels] } : {}),
});

/**
 * With no flag filter the grid lists the items the Bookmarked strip does not; with one, the grid alone lists exactly
 * the matching items and the strip is hidden.
 */
export const getGalleryGridListingMarks = (filter: GalleryMarksFilter): GalleryListingMarks =>
  canonicalizeGalleryListingMarks(
    filter.flags.length > 0
      ? { ...getRatingAndLabelMarks(filter), flags: [...filter.flags] }
      : { ...getRatingAndLabelMarks(filter), starred: false }
  );

/** Null while a flag filter hides the strip. */
export const getGalleryStripListingMarks = (filter: GalleryMarksFilter): GalleryListingMarks | null =>
  filter.flags.length > 0
    ? null
    : canonicalizeGalleryListingMarks({ ...getRatingAndLabelMarks(filter), starred: true });

/** The filter a reveal lands in: the current one where the item's marks still pass it, else cleared to list it. */
export const getGalleryRevealMarksFilter = (filter: GalleryMarksFilter, marks: GalleryMarks): GalleryMarksFilter => {
  const keepsRatingAndLabels = matchesGalleryListingMarks(getRatingAndLabelMarks(filter), marks);
  const base = keepsRatingAndLabels ? filter : { ...filter, colorLabels: [], rating: null };

  if (filter.flags.includes(marks.flag)) {
    return base;
  }

  // A bookmarked item sits in the strip, which the grid's index does not count; the bookmarked flag lists it there.
  return { ...base, flags: marks.flag === 'pick' ? ['pick'] : [] };
};

/** Equal for filters that list the same items. */
export const galleryMarksFilterKey = (filter: GalleryMarksFilter): string =>
  JSON.stringify(getGalleryGridListingMarks(filter));
