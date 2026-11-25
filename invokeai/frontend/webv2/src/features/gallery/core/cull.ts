import {
  applyGalleryMarksPatch,
  chooseGalleryColorLabel,
  getGalleryGridListingMarks,
  getGalleryItemMarks,
  getGalleryStripListingMarks,
  matchesGalleryListingMarks,
  setGalleryFlag,
  setGalleryRating,
  stepGalleryRating,
  type GalleryColorLabel,
  type GalleryItemMarkFields,
  type GalleryKnownMarks,
  type GalleryMarkFlag,
  type GalleryMarksFilter,
  type GalleryMarksPatch,
  type GalleryRating,
} from './marks';

export interface GalleryCullMode {
  autoAdvance: boolean;
  enabled: boolean;
}

/** Values from before Cull mode read as off, with auto-advance on. */
export const getGalleryCullMode = (values: Readonly<Record<string, unknown>>): GalleryCullMode => ({
  autoAdvance: values.cullAutoAdvance !== false,
  enabled: values.cullMode === true,
});

export const getGalleryCullModeValues = (
  patch: Partial<GalleryCullMode>
): { cullAutoAdvance?: boolean; cullMode?: boolean } => ({
  ...(patch.autoAdvance === undefined ? {} : { cullAutoAdvance: patch.autoAdvance }),
  ...(patch.enabled === undefined ? {} : { cullMode: patch.enabled }),
});

export const GALLERY_CULL_TOGGLE_COMMAND_ID = 'cull.toggleMode';

export type GalleryCullAction =
  | { kind: 'flag'; flag: GalleryMarkFlag }
  | { kind: 'label'; colorLabel: GalleryColorLabel }
  | { kind: 'rating'; rating: GalleryRating }
  | { kind: 'step'; step: -1 | 1 };

/**
 * The marking commands, by hotkey id: saved custom bindings are keyed by these. The Workbench hotkey catalog lists their
 * keys under the same ids (pinned by its test). Purple has no key and stays a pointer choice.
 */
export const GALLERY_CULL_COMMANDS = [
  { action: { kind: 'rating', rating: 1 }, id: 'cull.rate1' },
  { action: { kind: 'rating', rating: 2 }, id: 'cull.rate2' },
  { action: { kind: 'rating', rating: 3 }, id: 'cull.rate3' },
  { action: { kind: 'rating', rating: 4 }, id: 'cull.rate4' },
  { action: { kind: 'rating', rating: 5 }, id: 'cull.rate5' },
  { action: { kind: 'rating', rating: 0 }, id: 'cull.clearRating' },
  { action: { kind: 'step', step: -1 }, id: 'cull.lowerRating' },
  { action: { kind: 'step', step: 1 }, id: 'cull.raiseRating' },
  { action: { colorLabel: 'red', kind: 'label' }, id: 'cull.labelRed' },
  { action: { colorLabel: 'yellow', kind: 'label' }, id: 'cull.labelYellow' },
  { action: { colorLabel: 'green', kind: 'label' }, id: 'cull.labelGreen' },
  { action: { colorLabel: 'blue', kind: 'label' }, id: 'cull.labelBlue' },
  { action: { flag: 'pick', kind: 'flag' }, id: 'cull.bookmark' },
  { action: { flag: 'reject', kind: 'flag' }, id: 'cull.reject' },
  { action: { flag: 'none', kind: 'flag' }, id: 'cull.clearFlag' },
] as const satisfies readonly { action: GalleryCullAction; id: string }[];

/** The hotkey mode Preview and the Gallery turn on while Cull mode is; the catalog's marking rows name it. */
export const GALLERY_CULL_HOTKEY_MODE = 'cull';

export type GalleryCullCommandId = (typeof GALLERY_CULL_COMMANDS)[number]['id'];

/** Ratings and flags set; labels toggle; steps start from the primary item's rating. */
export const getGalleryCullPatch = (
  action: GalleryCullAction,
  targets: readonly GalleryKnownMarks[],
  primary: GalleryKnownMarks | undefined
): GalleryMarksPatch => {
  switch (action.kind) {
    case 'flag':
      return setGalleryFlag(action.flag);
    case 'label':
      return chooseGalleryColorLabel(targets, action.colorLabel);
    case 'rating':
      return setGalleryRating(action.rating);
    case 'step':
      return stepGalleryRating(primary, action.step);
  }
};

/** Steps never advance; Shift does the opposite of the switch for that press; a selection of several stays. */
export const shouldGalleryCullAdvance = (
  action: GalleryCullAction,
  { autoAdvance, inverted, targetCount }: { autoAdvance: boolean; inverted: boolean; targetCount: number }
): boolean => action.kind !== 'step' && targetCount === 1 && autoAdvance !== inverted;

/** The hotkey runtime hands a Shift-accepting command `{ shiftKey }`; the palette hands it nothing. */
export const isGalleryCullPressInverted = (press: unknown): boolean =>
  typeof press === 'object' && press !== null && (press as { shiftKey?: unknown }).shiftKey === true;

/**
 * Whether the marked item drops out of the listing: it matches neither the Bookmarked strip nor the grid. Moving between
 * the two keeps it listed.
 */
export const doesGalleryCullMarkLeaveListing = (
  item: GalleryItemMarkFields,
  patch: GalleryMarksPatch,
  filter: GalleryMarksFilter
): boolean => {
  const marks = getGalleryItemMarks(applyGalleryMarksPatch(item, patch));
  const strip = getGalleryStripListingMarks(filter);

  return (
    !matchesGalleryListingMarks(getGalleryGridListingMarks(filter), marks) &&
    (strip === null || !matchesGalleryListingMarks(strip, marks))
  );
};

/**
 * Where the selection goes after a marking key with one target: on, or onto the item that takes the place of one that
 * left the listing. The last item stays when it stays listed, and gives way to the one before it when it leaves.
 */
export const getGalleryCullStep = ({
  advance,
  hasNext,
  hasPrevious,
  leaves,
}: {
  advance: boolean;
  hasNext: boolean;
  hasPrevious: boolean;
  leaves: boolean;
}): 'next' | 'previous' | null =>
  !advance && !leaves ? null : hasNext ? 'next' : leaves && hasPrevious ? 'previous' : null;
