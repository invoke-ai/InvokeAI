import type {
  GalleryColorLabelFilter,
  GalleryMarkFlag,
  GalleryRating,
  GalleryRatingOperator,
} from '@features/gallery/core/marks';

import { Box, HStack, Separator } from '@chakra-ui/react';
import {
  chooseGalleryFilterRating,
  cycleGalleryFilterRatingOperator,
  EMPTY_GALLERY_MARKS_FILTER,
  GALLERY_COLOR_LABELS,
  isGalleryMarksFilterActive,
  toggleGalleryFilterColorLabel,
  toggleGalleryFilterFlag,
} from '@features/gallery/core/marks';
import { Button, IconButton, ToggleIconButton } from '@platform/ui/Button';
import { BookmarkIcon, CircleDashedIcon, FlagIcon, ListFilterIcon, type LucideIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { GalleryColorLabelDot, getGalleryColorLabelName } from './GalleryItemMarks';
import { GalleryRatingGroup } from './GalleryMarksControl';
import { useGalleryWidget } from './GalleryWidgetContext';

/** A ranked similarity result keeps no marks filter. */
const useIsMarksFilterUnavailable = (): boolean => useGalleryWidget().gallery.semanticImageQuery !== null;

/**
 * Shows the marks filter row; inert but focusable (its tooltip says why) under a ranked similarity result. A dot says a
 * filter applies while the row is hidden.
 */
export const GalleryMarksFilterToggle = () => {
  const { t } = useTranslation();
  const { actions, gallery } = useGalleryWidget();
  const isUnavailable = useIsMarksFilterUnavailable();
  const isOpen = gallery.settings.marksFilterRowOpen;
  const isFiltering = isGalleryMarksFilterActive(gallery.marksFilter) && !isUnavailable;
  const handleCheckedChange = useCallback(
    (checked: boolean) => {
      if (!isUnavailable) {
        actions.updateSettings({ marksFilterRowOpen: checked });
      }
    },
    [actions, isUnavailable]
  );
  const label = t('widgets.gallery.marksFilter.toggle');

  return (
    <Box flexShrink={0} position="relative">
      <ToggleIconButton
        aria-disabled={isUnavailable || undefined}
        checked={isOpen && !isUnavailable}
        data-gallery-marks-filter-toggle
        icon={ListFilterIcon}
        label={label}
        size="md"
        tooltip={
          isUnavailable
            ? t('widgets.gallery.marksFilter.unavailable')
            : isFiltering && !isOpen
              ? t('widgets.gallery.marksFilter.activeTooltip')
              : label
        }
        onCheckedChange={handleCheckedChange}
      />
      {isFiltering && !isOpen ? (
        <Box
          aria-hidden="true"
          bg="accent.solid"
          boxSize="1.5"
          data-gallery-marks-filter-active=""
          insetInlineEnd="1"
          pointerEvents="none"
          position="absolute"
          rounded="full"
          top="1"
        />
      ) : null}
    </Box>
  );
};

const FLAG_FILTERS: readonly { flag: GalleryMarkFlag; icon: LucideIcon; labelKey: string }[] = [
  { flag: 'pick', icon: BookmarkIcon, labelKey: 'widgets.gallery.marksFilter.bookmarked' },
  { flag: 'none', icon: CircleDashedIcon, labelKey: 'widgets.gallery.marksFilter.unflagged' },
  { flag: 'reject', icon: FlagIcon, labelKey: 'widgets.gallery.marksFilter.rejected' },
];

const RATING_OPERATOR_LABEL_KEYS = {
  atLeast: 'widgets.gallery.marksFilter.ratingAtLeast',
  atMost: 'widgets.gallery.marksFilter.ratingAtMost',
  exactly: 'widgets.gallery.marksFilter.ratingExactly',
} as const satisfies Record<GalleryRatingOperator, string>;

const RATING_STAR_LABEL_KEYS = {
  atLeast: 'widgets.gallery.marksFilter.ratedAtLeast',
  atMost: 'widgets.gallery.marksFilter.ratedAtMost',
  exactly: 'widgets.gallery.marksFilter.ratedExactly',
} as const satisfies Record<GalleryRatingOperator, string>;

const FlagFilterToggle = ({ flag, icon, label }: { flag: GalleryMarkFlag; icon: LucideIcon; label: string }) => {
  const { actions, gallery } = useGalleryWidget();
  const handleCheckedChange = useCallback(
    () => actions.setMarksFilter(toggleGalleryFilterFlag(gallery.marksFilter, flag)),
    [actions, flag, gallery.marksFilter]
  );

  return (
    <ToggleIconButton
      checked={gallery.marksFilter.flags.includes(flag)}
      data-marks-filter-flag={flag}
      icon={icon}
      label={label}
      size="xs"
      onCheckedChange={handleCheckedChange}
    />
  );
};

const FILTER_STAR_SIZE = { glyph: '3', h: '5', w: '5' } as const;

const RatingFilter = () => {
  const { t } = useTranslation();
  const { actions, gallery } = useGalleryWidget();
  const filter = gallery.marksFilter;
  const valueText = useCallback(
    (count: number) => t(RATING_STAR_LABEL_KEYS[filter.ratingOperator], { count }),
    [filter.ratingOperator, t]
  );
  const handleChoose = useCallback(
    (rating: GalleryRating) => actions.setMarksFilter(chooseGalleryFilterRating(filter, rating)),
    [actions, filter]
  );

  return (
    <GalleryRatingGroup
      label={t('widgets.gallery.marks.rating')}
      starSize={FILTER_STAR_SIZE}
      value={filter.rating ?? 0}
      valueText={valueText}
      onChoose={handleChoose}
    />
  );
};

const LabelFilterToggle = ({ colorLabel }: { colorLabel: Exclude<GalleryColorLabelFilter, 'none'> }) => {
  const { t } = useTranslation();
  const { actions, gallery } = useGalleryWidget();
  const isChecked = gallery.marksFilter.colorLabels.includes(colorLabel);
  const label = t('widgets.gallery.marks.colorLabel', { label: getGalleryColorLabelName(t, colorLabel) });
  const handleClick = useCallback(
    () => actions.setMarksFilter(toggleGalleryFilterColorLabel(gallery.marksFilter, colorLabel)),
    [actions, colorLabel, gallery.marksFilter]
  );

  return (
    <IconButton
      aria-label={label}
      aria-pressed={isChecked}
      data-marks-filter-label={colorLabel}
      minW="5"
      size="xs"
      tooltip={label}
      variant="ghost"
      onClick={handleClick}
    >
      <GalleryColorLabelDot boxSize="2" checked={isChecked} colorLabel={colorLabel} />
    </IconButton>
  );
};

/** Flags, rating and labels AND-combine; within flags or labels, any chosen one matches. */
export const GalleryMarksFilterRow = () => {
  const { t } = useTranslation();
  const { actions, gallery } = useGalleryWidget();
  const filter = gallery.marksFilter;
  const isUnavailable = useIsMarksFilterUnavailable();
  const handleCycleOperator = useCallback(
    () => actions.setMarksFilter(cycleGalleryFilterRatingOperator(filter)),
    [actions, filter]
  );
  const handleClear = useCallback(() => actions.setMarksFilter(EMPTY_GALLERY_MARKS_FILTER), [actions]);

  if (!gallery.settings.marksFilterRowOpen || isUnavailable) {
    return null;
  }

  const operatorLabel = t(RATING_OPERATOR_LABEL_KEYS[filter.ratingOperator]);

  return (
    <HStack
      aria-label={t('widgets.gallery.marksFilter.label')}
      columnGap="1"
      data-gallery-marks-filter-row=""
      flexWrap="wrap"
      minW="0"
      role="group"
      rowGap="1"
    >
      <HStack gap="0.5">
        {FLAG_FILTERS.map(({ flag, icon, labelKey }) => (
          <FlagFilterToggle key={flag} flag={flag} icon={icon} label={t(labelKey)} />
        ))}
      </HStack>
      <Separator h="4" orientation="vertical" />
      <HStack gap="0">
        <RatingFilter />
        <Button
          aria-label={t('widgets.gallery.marksFilter.ratingOperator', { operator: operatorLabel })}
          color="fg.muted"
          data-marks-filter-operator={filter.ratingOperator}
          fontSize="sm"
          px="1.5"
          size="xs"
          variant="ghost"
          onClick={handleCycleOperator}
        >
          {operatorLabel}
        </Button>
      </HStack>
      <Separator h="4" orientation="vertical" />
      <HStack gap="0">
        {GALLERY_COLOR_LABELS.map((colorLabel) => (
          <LabelFilterToggle key={colorLabel} colorLabel={colorLabel} />
        ))}
      </HStack>
      <Button
        color="fg.muted"
        disabled={!isGalleryMarksFilterActive(filter)}
        ms="auto"
        size="xs"
        variant="ghost"
        onClick={handleClear}
      >
        {t('widgets.gallery.marksFilter.clear')}
      </Button>
    </HStack>
  );
};
