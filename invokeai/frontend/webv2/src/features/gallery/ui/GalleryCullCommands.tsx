import type { GalleryItem, GalleryItemKey, GalleryItemRef } from '@features/gallery/core/items';
import type { GalleryKnownMarks, GalleryMarksFilter, GalleryMarksPatch } from '@features/gallery/core/marks';

import {
  doesGalleryCullMarkLeaveListing,
  GALLERY_CULL_COMMANDS,
  GALLERY_CULL_HOTKEY_MODE,
  GALLERY_CULL_TOGGLE_COMMAND_ID,
  getGalleryCullPatch,
  getGalleryCullStep,
  isGalleryCullPressInverted,
  shouldGalleryCullAdvance,
  type GalleryCullAction,
} from '@features/gallery/core/cull';
import { toGalleryItemKey } from '@features/gallery/core/items';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useEffectEvent, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { GalleryWidgetRuntime } from './GalleryUiContext';

import { getGalleryCullFeedback, type GalleryCullFeedback } from './galleryCullFeedback';
import { GalleryAnnouncer } from './GalleryLoadError';

/** What a cull key marks, read at the press. */
export interface GalleryCullTarget {
  refs: readonly GalleryItemRef[];
  /** Aligned with `refs`. */
  marks: readonly GalleryKnownMarks[];
  /** Where `[` and `]` step from. */
  primary: GalleryKnownMarks | undefined;
  /** The single target as listed, so a mark can tell it leaves the listing; null for several, or one not loaded. */
  item: GalleryItem | null;
}

export interface GalleryCullStepper<Entry> {
  /** The gallery item one step from `fromKey`, past any session; null at the end or when `fromKey` is not shown. */
  peek: (fromKey: GalleryItemKey, direction: 'left' | 'right') => Entry | null;
  /** Selects the entry as an arrow would. */
  go: (entry: Entry, direction: 'left' | 'right') => void;
}

export interface GalleryCullCommandsProps<Entry> {
  autoAdvance: boolean;
  enabled: boolean;
  /** The listing's marks filter; null for a ranking, which keeps every mark. */
  marksFilter: GalleryMarksFilter | null;
  runtime: {
    commands: Pick<GalleryWidgetRuntime['commands'], 'register'>;
    hotkeys: Pick<GalleryWidgetRuntime['hotkeys'], 'activateMode'>;
  };
  stepper: GalleryCullStepper<Entry>;
  /** Null while nothing is shown to mark. */
  target: GalleryCullTarget | null;
  onFeedback?: (feedback: GalleryCullFeedback) => void;
  onMark: (refs: GalleryItemRef[], patch: GalleryMarksPatch) => void;
  onToggle: () => void;
}

/**
 * Cull mode's commands for one surface: the toggle always; while the mode is on, the mode itself and the marking keys.
 * Remount it by mode and language. A press moves on before it marks, from the listing as it was, so a mark that takes
 * the item out of the listing cannot lose the place.
 */
export const GalleryCullCommands = <Entry,>({
  autoAdvance,
  enabled,
  marksFilter,
  runtime,
  stepper,
  target,
  onFeedback,
  onMark,
  onToggle,
}: GalleryCullCommandsProps<Entry>) => {
  const { t } = useTranslation();
  const [announcement, setAnnouncement] = useState<{ text: string; token: number } | null>(null);

  const toggle = useEffectEvent(() => onToggle());
  const mark = useEffectEvent((action: GalleryCullAction, press: unknown) => {
    const refs = target?.refs ?? [];
    const patch = target && refs.length > 0 ? getGalleryCullPatch(action, target.marks, target.primary) : null;
    const [single] = refs.length === 1 ? refs : [];
    const item = target?.item ?? null;

    if (patch && single) {
      const key = toGalleryItemKey(single);
      const leaves = item !== null && marksFilter !== null && doesGalleryCullMarkLeaveListing(item, patch, marksFilter);
      const next = stepper.peek(key, 'right');
      const previous = stepper.peek(key, 'left');
      const step = getGalleryCullStep({
        advance: shouldGalleryCullAdvance(action, {
          autoAdvance,
          inverted: isGalleryCullPressInverted(press),
          targetCount: 1,
        }),
        hasNext: next !== null,
        hasPrevious: previous !== null,
        leaves,
      });

      if (step === 'next' && next !== null) {
        stepper.go(next, 'right');
      } else if (step === 'previous' && previous !== null) {
        stepper.go(previous, 'left');
      }
    }

    if (patch) {
      onMark([...refs], patch);
    }

    const feedback = getGalleryCullFeedback(t, patch, refs.length);

    if (feedback) {
      setAnnouncement((current) => ({ text: feedback.announcement, token: (current?.token ?? 0) + 1 }));
      onFeedback?.(feedback);
    }
  });

  /* eslint-disable react-hooks/rules-of-hooks -- useMountEffect is the repository's explicit useEffect wrapper */
  useMountEffect(() => {
    const disposers = [
      runtime.commands.register({
        handler: () => toggle(),
        id: GALLERY_CULL_TOGGLE_COMMAND_ID,
        title: t('widgets.gallery.cull.toggle'),
      }),
      ...(enabled
        ? [
            runtime.hotkeys.activateMode({ mode: GALLERY_CULL_HOTKEY_MODE }),
            ...GALLERY_CULL_COMMANDS.map(({ action, id }) =>
              runtime.commands.register({ handler: (press) => mark(action, press), id, title: id })
            ),
          ]
        : []),
    ];

    return () => disposers.forEach((dispose) => dispose());
  });
  /* eslint-enable react-hooks/rules-of-hooks */

  // Alternating regions make a repeated announcement a change in an existing region, so each is read.
  return enabled
    ? [0, 1].map((parity) => (
        <GalleryAnnouncer
          key={parity}
          message={announcement && announcement.token % 2 === parity ? announcement.text : ''}
        />
      ))
    : null;
};
