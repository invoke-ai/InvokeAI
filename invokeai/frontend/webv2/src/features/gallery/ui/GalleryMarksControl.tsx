import type {
  GalleryColorLabel,
  GalleryKnownMarks,
  GalleryMarksPatch,
  GalleryRating,
} from '@features/gallery/core/marks';
import type { TFunction } from 'i18next';

import {
  Box,
  Flex,
  HStack,
  Icon,
  Menu,
  Popover,
  Portal,
  RatingGroup,
  Text,
  type SystemStyleObject,
} from '@chakra-ui/react';
import {
  chooseGalleryColorLabel,
  chooseGalleryRating,
  GALLERY_COLOR_LABELS,
  GALLERY_MAX_RATING,
  GALLERY_RATINGS,
  getSharedGalleryMarks,
  setGalleryRating,
  stepGalleryRating,
  toggleGalleryBookmark,
  toggleGalleryReject,
  toGalleryRating,
} from '@features/gallery/core/marks';
import { Button, IconButton } from '@platform/ui/Button';
import { PopoverContent } from '@platform/ui/Popover';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { ChevronDownIcon } from 'lucide-react';
import { useCallback, useMemo, useState, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import {
  GalleryBookmarkGlyph,
  GalleryColorLabelDot,
  GalleryRatingStar,
  GalleryRejectGlyph,
  getGalleryColorLabelName,
} from './GalleryItemMarks';

export type GalleryMarksControlSize = 'sm' | 'md' | 'lg';

export type GalleryMarksControlPresentation = 'menu' | 'popover' | 'toolbar';

export interface GalleryMarksControlProps {
  /** Each target's known marks; a part is on only when every target shares its value. */
  targets: readonly GalleryKnownMarks[];
  /** The patch the chosen part makes, from the marks semantics. */
  onChange: (patch: GalleryMarksPatch) => void;
  /**
   * The primary item's marks, where rating steps start. Without one, a step up starts from the highest rating and a step
   * down from the lowest, so no target moves against the step.
   */
  primary?: GalleryKnownMarks;
  /**
   * `menu`: the compact row as context menu items; `popover`: the same row as buttons and radios; `toolbar`: a bar's
   * toolbar at `size`, with separate label dots.
   */
  presentation?: GalleryMarksControlPresentation;
  /** The toolbar's size; menus and popovers are compact. */
  size?: GalleryMarksControlSize;
  disabled?: boolean;
}

interface Metrics {
  button: 'md' | 'sm' | 'xs';
  dot: string;
  glyph: string;
  partW: string;
  star: GalleryRatingGroupProps['starSize'];
}

/**
 * The menu and the popover draw the same compact row from these, so they read as one control. Parts are 24px tiles
 * without a pressed fill (the glyph's fill carries the state); the pointer tints a flag's tile as a ghost button's hover,
 * and the keyboard rings the part it is on: the popover's focus, the menu's highlight once a key has moved it.
 */
const COMPACT = {
  dot: '4',
  glyph: '3.5',
  part: '6',
  star: { glyph: '3.5', h: '6', w: '4.5' },
} as const;

const COMPACT_KEYBOARD = '&:focus-visible, [data-marks-keyboard] &[data-highlighted]';

/**
 * A menu highlights the item under the pointer too, and keeps it highlighted as the pointer leaves, so the highlight
 * alone cannot tell the keyboard's: a key press in the menu marks the row, and the pointer moving unmarks it.
 */
const trackMenuKeyboard = (row: HTMLDivElement | null) => {
  const menu = row?.closest<HTMLElement>('[role="menu"]');

  if (!row || !menu) {
    return undefined;
  }

  const mark = () => row.setAttribute('data-marks-keyboard', '');
  const unmark = () => row.removeAttribute('data-marks-keyboard');

  menu.addEventListener('keydown', mark, true);
  menu.addEventListener('pointermove', unmark);

  return () => {
    menu.removeEventListener('keydown', mark, true);
    menu.removeEventListener('pointermove', unmark);
  };
};

const COMPACT_RING: SystemStyleObject = { outline: '2px solid', outlineColor: 'accent.solid', outlineOffset: '-2px' };

// Every value the menu item recipe sets is restated, so both presentations start from the same box.
const COMPACT_PART_CSS: SystemStyleObject = {
  alignItems: 'center',
  bg: 'transparent',
  borderRadius: 'l2',
  color: 'fg.muted',
  cursor: 'pointer',
  display: 'flex',
  flex: '0 0 auto',
  gap: '0',
  h: COMPACT.part,
  justifyContent: 'center',
  minH: '0',
  minW: '0',
  outline: 'none',
  p: '0',
  _disabled: { cursor: 'not-allowed', opacity: 0.4 },
};

const COMPACT_FLAG_CSS: SystemStyleObject = {
  ...COMPACT_PART_CSS,
  w: COMPACT.part,
  _hover: { bg: 'bg.hover' },
  [COMPACT_KEYBOARD]: COMPACT_RING,
};

// The block tints under the pointer as the flags do, while its stars preview the hovered rating.
const COMPACT_RATING_CSS: SystemStyleObject = {
  ...COMPACT_PART_CSS,
  px: '0.5',
  w: 'auto',
  _hover: { bg: 'bg.hover' },
  // The stars dim themselves.
  _disabled: { cursor: 'not-allowed', opacity: 1 },
  [COMPACT_KEYBOARD]: COMPACT_RING,
  '&:has([data-part=item]:focus-visible)': COMPACT_RING,
  '& [data-part=item]:focus-visible': { outline: 'none' },
};

const TOOLBAR_METRICS: Record<GalleryMarksControlSize, Metrics> = {
  lg: { button: 'md', dot: '3.5', glyph: '4.5', partW: '9', star: { glyph: '4.5', h: '7', w: '9' } },
  md: { button: 'sm', dot: '2.5', glyph: '3.5', partW: '6', star: { glyph: '3.5', h: '6', w: '6' } },
  sm: { button: 'xs', dot: '2', glyph: '3', partW: '5', star: { glyph: '3', h: '5', w: '5' } },
};

const TOOLBAR_RATING_CSS: SystemStyleObject = { borderRadius: 'l2', _hover: { bg: 'bg.hover' } };

// Above the row in a menu, where the next item sits right below; elsewhere the platform's placement, as the buttons'.
export interface GalleryRatingGroupProps {
  disabled?: boolean;
  /** Names the group. */
  label: string;
  /** The targets' ratings differ, so `value` is 0 and the arrow keys step through `onStep`. */
  mixed?: boolean;
  /** The chosen rating; choosing the shown rating again passes it again, so the caller can clear it. */
  onChoose: (rating: GalleryRating) => void;
  onStep?: (step: -1 | 1) => void;
  /** In a menu row, which is the keyboard target: the stars only take the pointer and stay out of the menu's tree. */
  pointerOnly?: boolean;
  css?: SystemStyleObject;
  starSize: { glyph: string; h: string; w: string };
  value: GalleryRating;
  /** Each star's name and tooltip. */
  valueText: (rating: number) => string;
}

// As the rating machine reads them.
const RATING_STEP_KEYS: Readonly<Record<string, -1 | 1>> = { ArrowDown: 1, ArrowLeft: -1, ArrowRight: 1, ArrowUp: -1 };

/** Five stars as one control: one tab stop, arrow keys step, and hovering previews the fill up to the pointer. */
export const GalleryRatingGroup = ({
  css,
  disabled = false,
  label,
  mixed = false,
  onChoose,
  onStep,
  pointerOnly = false,
  starSize,
  value,
  valueText,
}: GalleryRatingGroupProps) => {
  // A click that clears keeps the pointer on its star; its hover preview would hide the clear until the pointer moved.
  const [clearedUnder, setClearedUnder] = useState<number | null>(null);
  const translations = useMemo(() => ({ ratingValueText: valueText }), [valueText]);
  const handleValueChange = useCallback(
    ({ value: next }: RatingGroup.ValueChangeDetails) => onChoose(toGalleryRating(next)),
    [onChoose]
  );
  const handleHoverChange = useCallback(
    ({ hoveredValue }: RatingGroup.HoverChangeDetails) =>
      setClearedUnder((current) => (current === hoveredValue ? current : null)),
    []
  );
  // The machine reports no change for its own value, so choosing it again is read from the click. The machine handles
  // the click in a microtask against the value it last rendered; answering after it keeps it from re-choosing the star.
  const chooseAgain = useCallback(
    (rating: GalleryRating) => {
      if (rating === value) {
        setClearedUnder(rating);
        queueMicrotask(() => onChoose(rating));
      }
    },
    [onChoose, value]
  );
  // The machine steps from the value it shows, which a mixed selection does not have; it skips a handled key.
  const stepMixed = useCallback(
    (event: ReactKeyboardEvent) => {
      const step = mixed && !disabled ? RATING_STEP_KEYS[event.key] : undefined;

      if (step && onStep) {
        event.preventDefault();
        onStep(step);
      }
    },
    [disabled, mixed, onStep]
  );

  return (
    <RatingGroup.Root
      aria-hidden={pointerOnly || undefined}
      count={GALLERY_MAX_RATING}
      css={css}
      data-mark-part="rating"
      data-mixed={mixed || undefined}
      disabled={disabled}
      translations={translations}
      value={value}
      onHoverChange={handleHoverChange}
      onKeyDownCapture={stepMixed}
      onValueChange={handleValueChange}
    >
      {pointerOnly ? null : <RatingGroup.Label srOnly>{label}</RatingGroup.Label>}
      <RatingGroup.Control>
        <RatingGroup.Context>
          {({ hoveredValue, hovering }) => {
            const shown = hovering && hoveredValue !== clearedUnder ? hoveredValue : value;

            return GALLERY_RATINGS.map((rating) => (
              <RatingStar
                key={rating}
                filled={rating <= shown}
                pointerOnly={pointerOnly}
                rating={rating}
                size={starSize}
                tooltip={valueText(rating)}
                onClick={chooseAgain}
              />
            ));
          }}
        </RatingGroup.Context>
      </RatingGroup.Control>
    </RatingGroup.Root>
  );
};

const RatingStar = ({
  filled,
  onClick,
  pointerOnly,
  rating,
  size,
  tooltip,
}: {
  filled: boolean;
  onClick: (rating: GalleryRating) => void;
  pointerOnly: boolean;
  rating: GalleryRating;
  size: GalleryRatingGroupProps['starSize'];
  tooltip: string;
}) => {
  const handleClick = useCallback(() => onClick(rating), [onClick, rating]);

  return (
    <RatingGroup.Item
      h={size.h}
      index={rating}
      // Not a radio in a menu row: the machine focuses the chosen radio after a click, which would take the menu's focus.
      role={pointerOnly ? 'none' : undefined}
      tabIndex={pointerOnly ? -1 : undefined}
      w={size.w}
      onClick={handleClick}
    >
      <Tooltip content={tooltip} openDelay={300} positioning={pointerOnly ? MENU_TOOLTIP_POSITIONING : undefined}>
        <Flex align="center" as="span" h="full" justify="center" w="full">
          <GalleryRatingStar boxSize={size.glyph} filled={filled} />
        </Flex>
      </Tooltip>
    </RatingGroup.Item>
  );
};

interface MarkPart {
  checked: boolean;
  label: string;
  patch: GalleryMarksPatch;
  value: string;
}

interface FlagPart extends MarkPart {
  render: (glyphSize: string) => ReactNode;
}

interface LabelPart extends MarkPart {
  colorLabel: GalleryColorLabel;
}

/** From the primary item's rating; without one, up from the highest and down from the lowest. */
const stepTargetsRating = (
  targets: readonly GalleryKnownMarks[],
  primary: GalleryKnownMarks | undefined,
  step: -1 | 1
): GalleryMarksPatch => {
  const ratings = targets.map((target) => target.rating ?? 0);

  return stepGalleryRating(
    primary ?? { rating: toGalleryRating(step > 0 ? Math.max(...ratings) : Math.min(...ratings)) },
    step
  );
};

const useMarkParts = (targets: readonly GalleryKnownMarks[]) => {
  const { t } = useTranslation();

  return useMemo(() => {
    const shared = getSharedGalleryMarks(targets);
    const flags: FlagPart[] = [
      {
        checked: shared.flag === 'pick',
        label: t('widgets.gallery.marks.bookmark'),
        patch: toggleGalleryBookmark(targets),
        render: (glyphSize) => <GalleryBookmarkGlyph boxSize={glyphSize} checked={shared.flag === 'pick'} />,
        value: 'bookmark',
      },
      {
        checked: shared.flag === 'reject',
        label: t('widgets.gallery.marks.reject'),
        patch: toggleGalleryReject(targets),
        render: (glyphSize) => <GalleryRejectGlyph boxSize={glyphSize} checked={shared.flag === 'reject'} />,
        value: 'reject',
      },
    ];
    const labels = GALLERY_COLOR_LABELS.map((colorLabel): LabelPart => ({
      checked: shared.colorLabel === colorLabel,
      colorLabel,
      label: t('widgets.gallery.marks.colorLabel', { label: getGalleryColorLabelName(t, colorLabel) }),
      patch: chooseGalleryColorLabel(targets, colorLabel),
      value: `label-${colorLabel}`,
    }));

    return {
      flags,
      labels,
      // Known ratings that differ; unknown ones beside unrated ones read as unrated.
      mixed: shared.rating === undefined && targets.some((target) => (target.rating ?? 0) > 0),
      rating: shared.rating ?? 0,
      sharedRating: shared.rating,
    };
  }, [t, targets]);
};

/** Decorative: a separator role would read as a menu separator inside a context menu. */
const GroupDivider = ({ mx }: { mx: string }) => (
  <Box aria-hidden="true" borderColor="border" borderInlineStartWidth="1px" flexShrink={0} h="4" mx={mx} />
);

const ToolbarFlagPart = ({
  disabled,
  metrics,
  onChange,
  part,
}: {
  disabled: boolean;
  metrics: Metrics;
  onChange: (patch: GalleryMarksPatch) => void;
  part: FlagPart;
}) => {
  const handleClick = useCallback(() => onChange(part.patch), [onChange, part.patch]);

  return (
    <IconButton
      aria-label={part.label}
      aria-pressed={part.checked}
      bg={part.checked ? 'bg.emphasized' : undefined}
      color={part.checked ? 'fg' : 'fg.muted'}
      data-mark-part={part.value}
      disabled={disabled}
      minW={metrics.partW}
      size={metrics.button}
      tooltip={part.label}
      variant="ghost"
      onClick={handleClick}
    >
      {part.render(metrics.glyph)}
    </IconButton>
  );
};

const ToolbarDotPart = ({
  disabled,
  metrics,
  onChange,
  part,
}: {
  disabled: boolean;
  metrics: Metrics;
  onChange: (patch: GalleryMarksPatch) => void;
  part: LabelPart;
}) => {
  const handleClick = useCallback(() => onChange(part.patch), [onChange, part.patch]);

  return (
    <IconButton
      aria-label={part.label}
      aria-pressed={part.checked}
      bg={part.checked ? 'bg.emphasized' : undefined}
      data-mark-part={part.value}
      disabled={disabled}
      minW={metrics.partW}
      size={metrics.button}
      tooltip={part.label}
      variant="ghost"
      onClick={handleClick}
    >
      <GalleryColorLabelDot boxSize={metrics.dot} checked={part.checked} colorLabel={part.colorLabel} />
    </IconButton>
  );
};

const PopoverFlagPart = ({
  disabled,
  onChange,
  part,
}: {
  disabled: boolean;
  onChange: (patch: GalleryMarksPatch) => void;
  part: FlagPart;
}) => {
  const handleClick = useCallback(() => onChange(part.patch), [onChange, part.patch]);

  return (
    <IconButton
      aria-label={part.label}
      aria-pressed={part.checked}
      css={COMPACT_FLAG_CSS}
      data-mark-part={part.value}
      disabled={disabled}
      tooltip={part.label}
      unstyled
      onClick={handleClick}
    >
      {part.render(COMPACT.glyph)}
    </IconButton>
  );
};

const MENU_TOOLTIP_POSITIONING = { placement: 'top' } as const;

/** A menu checkbox item: Zag highlights it with `aria-activedescendant`, so the tooltip shows on hover only. */
const MenuFlagPart = ({
  disabled,
  onChange,
  part,
}: {
  disabled: boolean;
  onChange: (patch: GalleryMarksPatch) => void;
  part: FlagPart;
}) => {
  const handleCheckedChange = useCallback(() => onChange(part.patch), [onChange, part.patch]);

  return (
    <Menu.CheckboxItem
      aria-label={part.label}
      checked={part.checked}
      closeOnSelect={false}
      css={COMPACT_FLAG_CSS}
      data-mark-part={part.value}
      disabled={disabled}
      value={`marks-${part.value}`}
      onCheckedChange={handleCheckedChange}
    >
      <Tooltip content={part.label} openDelay={300} positioning={MENU_TOOLTIP_POSITIONING}>
        <Flex align="center" h="full" justify="center" w="full">
          {part.render(COMPACT.glyph)}
        </Flex>
      </Tooltip>
    </Menu.CheckboxItem>
  );
};

/**
 * Overlapping label circles, each its own target: a ring in the surface color (`--gallery-marks-surface`) parts it from
 * the one beneath. The chosen label rises with a ring in the text color, so it never reads by hue alone; a hovered or
 * focused one comes to the front.
 */
const STACK_PART_CSS: SystemStyleObject = {
  ...COMPACT_PART_CSS,
  position: 'relative',
  w: COMPACT.dot,
  // With the 2px ring drawn over it, each circle hides about a third of the one before.
  '&:not(:first-of-type)': { ms: '-3px' },
  '& > [data-label-dot]': {
    boxShadow: '0 0 0 2px var(--gallery-marks-surface)',
    transitionDuration: 'faster',
    transitionProperty: 'box-shadow',
    _motionReduce: { transitionDuration: '0s' },
  },
  '&[aria-pressed=true], &[aria-checked=true]': {
    zIndex: 1,
    '& > [data-label-dot]': {
      boxShadow: '0 0 0 1.5px var(--gallery-marks-surface), 0 0 0 3px {colors.fg}',
    },
  },
  '&:hover, &[data-highlighted], &:focus-visible': { zIndex: 2 },
  '&:focus-visible > [data-label-dot], [data-marks-keyboard] &[data-highlighted] > [data-label-dot]': {
    outline: '2px solid',
    outlineColor: 'accent.solid',
    outlineOffset: '2px',
  },
};

const PopoverStackPart = ({
  disabled,
  onChange,
  part,
}: {
  disabled: boolean;
  onChange: (patch: GalleryMarksPatch) => void;
  part: LabelPart;
}) => {
  const handleClick = useCallback(() => onChange(part.patch), [onChange, part.patch]);

  return (
    <IconButton
      aria-label={part.label}
      aria-pressed={part.checked}
      css={STACK_PART_CSS}
      data-mark-part={part.value}
      disabled={disabled}
      tooltip={part.label}
      unstyled
      onClick={handleClick}
    >
      <GalleryColorLabelDot boxSize={COMPACT.dot} colorLabel={part.colorLabel} data-label-dot="" />
    </IconButton>
  );
};

const MenuStackPart = ({
  disabled,
  onChange,
  part,
}: {
  disabled: boolean;
  onChange: (patch: GalleryMarksPatch) => void;
  part: LabelPart;
}) => {
  const handleCheckedChange = useCallback(() => onChange(part.patch), [onChange, part.patch]);

  return (
    <Menu.CheckboxItem
      aria-label={part.label}
      checked={part.checked}
      closeOnSelect={false}
      css={STACK_PART_CSS}
      data-mark-part={part.value}
      disabled={disabled}
      value={`marks-${part.value}`}
      onCheckedChange={handleCheckedChange}
    >
      <Tooltip content={part.label} openDelay={300} positioning={MENU_TOOLTIP_POSITIONING}>
        <GalleryColorLabelDot boxSize={COMPACT.dot} colorLabel={part.colorLabel} data-label-dot="" />
      </Tooltip>
    </Menu.CheckboxItem>
  );
};

/**
 * The rating as one menu row. The menu keeps focus and moves its highlight with the arrow keys, so while this row is
 * highlighted, Left and Right step the rating and 0–5 set it; the stars inside take the pointer.
 */
const MenuRatingPart = ({
  disabled,
  mixed,
  name,
  onChange,
  onChoose,
  onStep,
  rating,
  sharedRating,
  valueText,
}: {
  disabled: boolean;
  mixed: boolean;
  /** The row's accessible name, which announces its value. */
  name: string;
  onChange: (patch: GalleryMarksPatch) => void;
  onChoose: (rating: GalleryRating) => void;
  onStep: (step: -1 | 1) => GalleryMarksPatch;
  rating: GalleryRating;
  /** The rating every target has; undefined when they differ or one is unknown. */
  sharedRating: GalleryRating | undefined;
  valueText: (rating: number) => string;
}) => {
  const { t } = useTranslation();
  // The menu's content holds focus, so its keydown reaches no item; a listener there runs before Zag's own handler.
  const listenForRatingKeys = useCallback(
    (row: HTMLDivElement | null) => {
      const menu = row?.closest<HTMLElement>('[role="menu"]');

      if (!row || !menu || disabled) {
        return undefined;
      }

      const handleKeyDown = (event: KeyboardEvent) => {
        if (
          event.defaultPrevented ||
          event.altKey ||
          event.ctrlKey ||
          event.metaKey ||
          menu.getAttribute('aria-activedescendant') !== row.id
        ) {
          return;
        }

        const patch =
          event.key === 'ArrowLeft'
            ? onStep(-1)
            : event.key === 'ArrowRight'
              ? onStep(1)
              : /^[0-5]$/.test(event.key)
                ? setGalleryRating(toGalleryRating(Number(event.key)))
                : null;

        if (patch) {
          event.preventDefault();
          if (patch.rating !== sharedRating) {
            onChange(patch);
          }
        }
      };

      menu.addEventListener('keydown', handleKeyDown);
      return () => menu.removeEventListener('keydown', handleKeyDown);
    },
    [disabled, onChange, onStep, sharedRating]
  );

  return (
    <Menu.Item
      ref={listenForRatingKeys}
      aria-description={t('widgets.gallery.marks.ratingKeysHint')}
      aria-label={name}
      closeOnSelect={false}
      css={COMPACT_RATING_CSS}
      data-mark-part="rating-row"
      disabled={disabled}
      value="marks-rating"
    >
      <GalleryRatingGroup
        disabled={disabled}
        label={t('widgets.gallery.marks.rating')}
        mixed={mixed}
        pointerOnly
        starSize={COMPACT.star}
        value={rating}
        valueText={valueText}
        onChoose={onChoose}
      />
    </Menu.Item>
  );
};

// The ring around stacked labels takes the color of the surface they sit on.
const SURFACE_CSS: Record<GalleryMarksControlPresentation, SystemStyleObject> = {
  menu: { '--gallery-marks-surface': 'var(--menu-bg)' },
  popover: { '--gallery-marks-surface': 'var(--popover-bg)' },
  toolbar: {},
};

/** `[bookmark][reject] | ★★★★★ | labels`, for one item or a selection. */
export const GalleryMarksControl = ({
  disabled = false,
  onChange,
  presentation = 'toolbar',
  primary,
  size = 'md',
  targets,
}: GalleryMarksControlProps) => {
  const { t } = useTranslation();
  const { flags, labels, mixed, rating, sharedRating } = useMarkParts(targets);
  const metrics = TOOLBAR_METRICS[size];
  const isMenu = presentation === 'menu';
  const isToolbar = presentation === 'toolbar';
  const isRoomy = presentation === 'toolbar' && size === 'lg';
  const valueText = useCallback((count: number) => t('widgets.gallery.marks.ratingSummary', { count }), [t]);
  const chooseRating = useCallback(
    (next: GalleryRating) => onChange(chooseGalleryRating(targets, next)),
    [onChange, targets]
  );
  const getStep = useCallback((step: -1 | 1) => stepTargetsRating(targets, primary, step), [primary, targets]);
  const step = useCallback((by: -1 | 1) => onChange(getStep(by)), [getStep, onChange]);
  const ratingName = t('widgets.gallery.marks.ratingMenuItem', {
    value: mixed
      ? t('widgets.gallery.marks.ratingsDiffer')
      : rating
        ? valueText(rating)
        : t('widgets.gallery.marks.unrated'),
  });
  const divider = <GroupDivider mx={isRoomy ? '1.5' : '1'} />;

  return (
    <HStack
      ref={isMenu ? trackMenuKeyboard : undefined}
      aria-label={t('widgets.gallery.marks.label')}
      css={SURFACE_CSS[presentation]}
      data-gallery-marks-control=""
      gap="0"
      role="group"
    >
      <HStack gap={isRoomy ? '0.5' : '0'}>
        {flags.map((part) =>
          isToolbar ? (
            <ToolbarFlagPart key={part.value} disabled={disabled} metrics={metrics} part={part} onChange={onChange} />
          ) : isMenu ? (
            <MenuFlagPart key={part.value} disabled={disabled} part={part} onChange={onChange} />
          ) : (
            <PopoverFlagPart key={part.value} disabled={disabled} part={part} onChange={onChange} />
          )
        )}
      </HStack>
      {divider}
      {isMenu ? (
        <MenuRatingPart
          disabled={disabled}
          mixed={mixed}
          name={ratingName}
          rating={rating}
          sharedRating={sharedRating}
          valueText={valueText}
          onChange={onChange}
          onChoose={chooseRating}
          onStep={getStep}
        />
      ) : (
        <GalleryRatingGroup
          css={isToolbar ? TOOLBAR_RATING_CSS : COMPACT_RATING_CSS}
          disabled={disabled}
          label={mixed ? ratingName : t('widgets.gallery.marks.rating')}
          mixed={mixed}
          starSize={isToolbar ? metrics.star : COMPACT.star}
          value={rating}
          valueText={valueText}
          onChoose={chooseRating}
          onStep={step}
        />
      )}
      {divider}
      {isToolbar ? (
        <HStack gap={isRoomy ? '0.5' : '0'}>
          {labels.map((part) => (
            <ToolbarDotPart key={part.value} disabled={disabled} metrics={metrics} part={part} onChange={onChange} />
          ))}
        </HStack>
      ) : (
        <HStack gap="0" px="1">
          {labels.map((part) =>
            isMenu ? (
              <MenuStackPart key={part.value} disabled={disabled} part={part} onChange={onChange} />
            ) : (
              <PopoverStackPart key={part.value} disabled={disabled} part={part} onChange={onChange} />
            )
          )}
        </HStack>
      )}
    </HStack>
  );
};

/**
 * "Rejected, 3 stars, Red label", or null when the targets share none of these. `withBookmark` adds "Bookmarked", for a
 * trigger with no bookmark toggle beside it.
 */
export const getGalleryMarksSummary = (
  t: TFunction,
  targets: readonly GalleryKnownMarks[],
  withBookmark = false
): string | null => {
  const { colorLabel, flag, rating } = getSharedGalleryMarks(targets);
  const parts = [
    withBookmark && flag === 'pick' ? t('widgets.gallery.marks.bookmarked') : null,
    flag === 'reject' ? t('widgets.gallery.marks.rejected') : null,
    rating ? t('widgets.gallery.marks.ratingSummary', { count: rating }) : null,
    colorLabel ? t('widgets.gallery.marks.colorLabel', { label: getGalleryColorLabelName(t, colorLabel) }) : null,
  ].filter((part): part is string => part !== null);

  return parts.length > 0 ? parts.join(', ') : null;
};

const POPOVER_POSITIONINGS = {
  'bottom-end': { placement: 'bottom-end' },
  'top-end': { placement: 'top-end' },
} as const;

export interface GalleryMarksPopoverProps extends Pick<
  GalleryMarksControlProps,
  'disabled' | 'onChange' | 'primary' | 'targets'
> {
  /** A bar along the bottom edge opens it upwards. */
  placement?: keyof typeof POPOVER_POSITIONINGS;
  /** The trigger's ghost-button size. */
  size?: 'sm' | 'xs';
  /** The trigger shows the bookmark too: set where no bookmark toggle sits beside it. */
  showsBookmark?: boolean;
}

/**
 * A trigger that shows the shared reject, rating and label, opening the marks control. Focus returns to the trigger as the
 * popover closes.
 */
export const GalleryMarksPopover = ({
  disabled = false,
  onChange,
  placement = 'bottom-end',
  primary,
  showsBookmark = false,
  size = 'sm',
  targets,
}: GalleryMarksPopoverProps) => {
  const { t } = useTranslation();
  const triggerIds = useTooltipTriggerIds();
  const { colorLabel, flag, rating } = getSharedGalleryMarks(targets);
  const summary = getGalleryMarksSummary(t, targets, showsBookmark);
  const label = summary
    ? t('widgets.gallery.marks.triggerWithSummary', { summary })
    : t('widgets.gallery.marks.trigger');

  return (
    <Popover.Root ids={triggerIds} lazyMount positioning={POPOVER_POSITIONINGS[placement]} unmountOnExit>
      <Tooltip content={label} ids={triggerIds}>
        <Popover.Trigger asChild>
          <Button
            aria-label={label}
            color={summary ? 'fg' : 'fg.muted'}
            data-gallery-marks-trigger=""
            disabled={disabled}
            flexShrink={0}
            gap="1"
            minW="0"
            px="1.5"
            size={size}
            variant="ghost"
          >
            {showsBookmark && flag === 'pick' ? (
              <GalleryBookmarkGlyph boxSize="3.5" checked data-gallery-marks-trigger-bookmark="" />
            ) : null}
            {flag === 'reject' ? (
              <GalleryRejectGlyph boxSize="3.5" checked data-gallery-marks-trigger-detail="" />
            ) : null}
            <GalleryRatingStar boxSize="3.5" filled={Boolean(rating)} />
            {rating ? (
              <Text as="span" data-gallery-marks-trigger-rating="" fontSize="sm" fontVariantNumeric="tabular-nums">
                {rating}
              </Text>
            ) : null}
            {colorLabel ? (
              <GalleryColorLabelDot boxSize="2" colorLabel={colorLabel} data-gallery-marks-trigger-detail="" />
            ) : null}
            <Icon as={ChevronDownIcon} boxSize="2.5" data-gallery-marks-trigger-detail="" />
          </Button>
        </Popover.Trigger>
      </Tooltip>
      <Portal>
        <Popover.Positioner>
          <PopoverContent p="1" w="auto">
            <GalleryMarksControl presentation="popover" primary={primary} targets={targets} onChange={onChange} />
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};
