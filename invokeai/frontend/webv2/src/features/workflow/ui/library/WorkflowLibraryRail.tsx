import type { WorkflowLibraryPlace, WorkflowTagCount } from '@features/workflow/data/libraryBrowseStore';
import type { WorkflowLibraryTab } from '@features/workflow/ui/workflowUiStore';
import type { LucideIcon } from 'lucide-react';

import { Box, Checkbox, createListCollection, HStack, Icon, Popover, Portal, Stack, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { PopoverContent } from '@platform/ui/Popover';
import { Scrollable } from '@platform/ui/Scrollable';
import { Select } from '@platform/ui/Select';
import { BookOpenIcon, ClockIcon, FolderIcon, LayoutTemplateIcon, UsersIcon } from 'lucide-react';
import { useCallback, useId, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * The library's left rail: the places it browses, then the current place's tags as any-of checkboxes. Counts are
 * the place's own, never narrowed by the search or the checked tags, so checking a tag never changes its neighbours.
 */

const RAIL_WIDTH = '12.25rem';

/**
 * The library's one layout breakpoint, on the dialog's container (`containerName: workflow-library-dialog`): below
 * 56rem (a window under about 960px) the rail, a usable card column and the 18rem detail rail no longer fit side by
 * side. There the places become a select in the toolbar, the tags a popover behind a Tags button that counts the
 * checked ones, the order and view controls one options menu, and the detail rail a bottom sheet
 * (`WorkflowDetailSheet`).
 */
export const WORKFLOW_LIBRARY_NARROW = '@container workflow-library-dialog (width < 56rem)';
/** The complement of `WORKFLOW_LIBRARY_NARROW`: the rail, centre column and detail rail side by side. */
export const WORKFLOW_LIBRARY_WIDE = '@container workflow-library-dialog (width >= 56rem)';
/** Controls that only the wide layout shows. */
export const WORKFLOW_LIBRARY_WIDE_ONLY_CSS = { [WORKFLOW_LIBRARY_NARROW]: { display: 'none' } } as const;

const TAG_HOVER = { bg: 'bg.muted' } as const;

const PLACES: ReadonlyArray<{ icon: LucideIcon; labelKey: string; value: WorkflowLibraryTab }> = [
  { icon: FolderIcon, labelKey: 'workflowLibrary.placeProject', value: 'project' },
  { icon: ClockIcon, labelKey: 'workflowLibrary.placeRecent', value: 'recent' },
  { icon: BookOpenIcon, labelKey: 'workflowLibrary.placeUser', value: 'user' },
  { icon: LayoutTemplateIcon, labelKey: 'workflowLibrary.placeDefault', value: 'default' },
  { icon: UsersIcon, labelKey: 'workflowLibrary.placeShared', value: 'shared' },
];

/** A place's name, as the rail lists it. */
export const getWorkflowLibraryPlaceLabelKey = (place: WorkflowLibraryTab): string =>
  PLACES.find((candidate) => candidate.value === place)!.labelKey;

/** The id of a place's label in the rail, which names the centre column showing that place. */
export const getWorkflowLibraryPlaceLabelId = (idBase: string, place: WorkflowLibraryTab) => `${idBase}-place-${place}`;

export interface WorkflowLibraryRailProps {
  /** Prefix for the place labels' ids; see `getWorkflowLibraryPlaceLabelId`. */
  idBase: string;
  place: WorkflowLibraryTab;
  /** Unfiltered totals per library place; an absent place shows no count. */
  placeTotals: Readonly<Partial<Record<WorkflowLibraryPlace, number>>>;
  /** In this project's count. */
  projectTotal: number;
  showShared: boolean;
  tagCounts: readonly WorkflowTagCount[];
  checkedTags: readonly string[];
  /** A sentence under the rail, for the place it describes. */
  footer?: string;
  onPlaceChange: (place: WorkflowLibraryTab) => void;
  onTagsChange: (tags: readonly string[]) => void;
}

const SectionLabel = ({ children, id }: { children: string; id: string }) => (
  <Text color="fg.muted" fontSize="xs" fontWeight="600" id={id} pb="1" px="2" textTransform="uppercase">
    {children}
  </Text>
);

// Localized labels wrap rather than clip, so every place and tag reads in full.
const LABEL_WRAP = { overflowWrap: 'anywhere', whiteSpace: 'normal' } as const;

const PlaceButton = ({
  count,
  icon,
  isCurrent,
  label,
  labelId,
  place,
  onSelect,
}: {
  count: string | undefined;
  icon: LucideIcon;
  labelId: string;
  isCurrent: boolean;
  label: string;
  place: WorkflowLibraryTab;
  onSelect: (place: WorkflowLibraryTab) => void;
}) => {
  const handleClick = useCallback(() => onSelect(place), [onSelect, place]);

  return (
    <Button
      aria-current={isCurrent ? 'page' : undefined}
      data-library-place={place}
      gap="2"
      h="auto"
      justifyContent="start"
      minH="7"
      py="1"
      variant={isCurrent ? 'subtle' : 'ghost'}
      w="full"
      onClick={handleClick}
    >
      <Icon aria-hidden as={icon} boxSize="3.5" flexShrink={0} />
      <Text css={LABEL_WRAP} flex="1" id={labelId} minW="0" textAlign="start">
        {label}
      </Text>
      {count === undefined ? null : (
        // On the current place's tinted background the muted tone falls under 4.5:1; the label's own colour holds.
        <Text
          as="span"
          color={isCurrent ? 'inherit' : 'fg.muted'}
          flexShrink={0}
          fontSize="xs"
          fontVariantNumeric="tabular-nums"
        >
          {count}
        </Text>
      )}
    </Button>
  );
};

const TagCheckbox = ({
  count,
  isChecked,
  isListed,
  tag,
  onToggle,
}: {
  count: string;
  isChecked: boolean;
  isListed: boolean;
  tag: string;
  onToggle: (tag: string, isChecked: boolean) => void;
}) => {
  const handleChange = useCallback(
    ({ checked }: { checked: boolean | 'indeterminate' }) => onToggle(tag, checked === true),
    [onToggle, tag]
  );

  return (
    <Checkbox.Root
      alignItems="center"
      checked={isChecked}
      data-library-tag={tag}
      data-unlisted={isListed ? undefined : ''}
      gap="2"
      minW="0"
      px="2"
      py="1"
      rounded="sm"
      size="sm"
      w="full"
      _hover={TAG_HOVER}
      onCheckedChange={handleChange}
    >
      <Checkbox.HiddenInput />
      <Checkbox.Control flexShrink={0}>
        <Checkbox.Indicator />
      </Checkbox.Control>
      <Checkbox.Label css={LABEL_WRAP} flex="1" fontSize="sm" fontWeight="normal" minW="0">
        {tag}
      </Checkbox.Label>
      <Text as="span" color="fg.muted" flexShrink={0} fontSize="xs" fontVariantNumeric="tabular-nums">
        {count}
      </Text>
    </Checkbox.Root>
  );
};

export interface WorkflowLibraryTagGroupProps {
  tagCounts: readonly WorkflowTagCount[];
  checkedTags: readonly string[];
  onTagsChange: (tags: readonly string[]) => void;
  /** Takes focus when the control holding it leaves and no listed tag remains to take it. */
  getFallbackFocus: () => HTMLElement | null | undefined;
}

/** The current place's tags as any-of checkboxes with Clear, shared by the rail and the narrow layout's popover. */
export const WorkflowLibraryTagGroup = ({
  checkedTags,
  getFallbackFocus,
  tagCounts,
  onTagsChange,
}: WorkflowLibraryTagGroupProps) => {
  const { i18n, t } = useTranslation();
  const numberFormat = useMemo(() => new Intl.NumberFormat(i18n.resolvedLanguage), [i18n.resolvedLanguage]);
  const groupRef = useRef<HTMLDivElement>(null);
  const tagsLabelId = useId();
  const tagsHintId = useId();
  // A checked tag the counts no longer list (a failed count, a deleted workflow) stays visible, so it can be cleared.
  const listedTags = useMemo(() => new Set(tagCounts.map(({ tag }) => tag)), [tagCounts]);
  /**
   * Controls that leave on this change (Clear with the last checked tag, an unlisted tag once unchecked) hand focus to
   * the first listed tag, or the fallback when no tag stays, rather than letting it fall to the page.
   */
  const keepFocus = useCallback(() => {
    (
      groupRef.current?.querySelector<HTMLElement>('[data-library-tag]:not([data-unlisted]) input[type="checkbox"]') ??
      getFallbackFocus()
    )?.focus();
  }, [getFallbackFocus]);
  const handleToggle = useCallback(
    (tag: string, isChecked: boolean) => {
      if (!isChecked && !listedTags.has(tag)) {
        keepFocus();
      }

      onTagsChange(isChecked ? [...checkedTags, tag] : checkedTags.filter((candidate) => candidate !== tag));
    },
    [checkedTags, keepFocus, listedTags, onTagsChange]
  );
  const handleClear = useCallback(() => {
    keepFocus();
    onTagsChange([]);
  }, [keepFocus, onTagsChange]);
  const tags = [...tagCounts, ...checkedTags.filter((tag) => !listedTags.has(tag)).map((tag) => ({ count: 0, tag }))];

  if (tags.length === 0) {
    return null;
  }

  return (
    <Stack ref={groupRef} aria-describedby={tagsHintId} aria-labelledby={tagsLabelId} gap="0" minW="0" role="group">
      <HStack justify="space-between" minW="0">
        <SectionLabel id={tagsLabelId}>{t('workflowLibrary.tags')}</SectionLabel>
        {checkedTags.length > 0 ? (
          <Button flexShrink={0} mb="1" size="xs" variant="plain" onClick={handleClear}>
            {t('workflowLibrary.clearTags')}
          </Button>
        ) : null}
      </HStack>
      {tags.map(({ count, tag }) => (
        <TagCheckbox
          key={tag}
          count={numberFormat.format(count)}
          isChecked={checkedTags.includes(tag)}
          isListed={listedTags.has(tag)}
          tag={tag}
          onToggle={handleToggle}
        />
      ))}
      <Text color="fg.muted" fontSize="xs" id={tagsHintId} pt="1.5" px="2">
        {t('workflowLibrary.tagsAnyHint')}
      </Text>
    </Stack>
  );
};

export const WorkflowLibraryRail = ({
  checkedTags,
  footer,
  idBase,
  place,
  placeTotals,
  projectTotal,
  showShared,
  tagCounts,
  onPlaceChange,
  onTagsChange,
}: WorkflowLibraryRailProps) => {
  const { i18n, t } = useTranslation();
  const numberFormat = useMemo(() => new Intl.NumberFormat(i18n.resolvedLanguage), [i18n.resolvedLanguage]);
  const formatTotal = (total: number | undefined) => (total === undefined ? undefined : numberFormat.format(total));
  const railRef = useRef<HTMLDivElement>(null);
  const placesLabelId = useId();
  const getCurrentPlace = useCallback(
    () => railRef.current?.querySelector<HTMLElement>('[data-library-place][aria-current="page"]'),
    []
  );

  return (
    <Stack
      ref={railRef}
      borderColor="border.subtle"
      borderEndWidth="1px"
      data-workflow-library-rail
      flexShrink={0}
      gap="0"
      minH="0"
      w={RAIL_WIDTH}
    >
      {/* Unlabelled: the nav and the tag group inside it name themselves. */}
      <Scrollable flex="1" minH="0">
        <Stack gap="4" minW="0" px="3" py="3">
          <Box as="nav" aria-labelledby={placesLabelId} minW="0">
            <SectionLabel id={placesLabelId}>{t('workflowLibrary.places')}</SectionLabel>
            <Stack as="ul" gap="0.5" listStyle="none" minW="0">
              {PLACES.filter((item) => item.value !== 'shared' || showShared).map((item) => (
                <Box as="li" key={item.value} minW="0">
                  <PlaceButton
                    count={formatTotal(item.value === 'project' ? projectTotal : placeTotals[item.value])}
                    icon={item.icon}
                    labelId={getWorkflowLibraryPlaceLabelId(idBase, item.value)}
                    isCurrent={place === item.value}
                    label={t(item.labelKey)}
                    place={item.value}
                    onSelect={onPlaceChange}
                  />
                </Box>
              ))}
            </Stack>
          </Box>
          <WorkflowLibraryTagGroup
            checkedTags={checkedTags}
            getFallbackFocus={getCurrentPlace}
            tagCounts={tagCounts}
            onTagsChange={onTagsChange}
          />
        </Stack>
      </Scrollable>
      {footer ? (
        <Text borderColor="border.subtle" borderTopWidth="1px" color="fg.muted" fontSize="xs" pb="3" px="3" pt="2.5">
          {footer}
        </Text>
      ) : null}
    </Stack>
  );
};

interface PlaceOption {
  count: string | undefined;
  label: string;
  value: WorkflowLibraryTab;
}

const PlaceOptionLabel = ({ count, label }: Pick<PlaceOption, 'count' | 'label'>) => (
  <Box as="span" alignItems="baseline" display="inline-flex" gap="2" minW="0" w="full">
    <Box as="span" flex="1" minW="0" truncate>
      {label}
    </Box>
    {count === undefined ? null : (
      <Text as="span" color="fg.muted" flexShrink={0} fontSize="xs" fontVariantNumeric="tabular-nums">
        {count}
      </Text>
    )}
  </Box>
);
const renderPlaceOption = (option: PlaceOption) => <PlaceOptionLabel count={option.count} label={option.label} />;

export type WorkflowLibraryPlaceSelectProps = Pick<
  WorkflowLibraryRailProps,
  'place' | 'placeTotals' | 'projectTotal' | 'showShared' | 'onPlaceChange'
>;

/** The narrow layout's places: a select at the start of the toolbar naming the current place and its count. */
export const WorkflowLibraryPlaceSelect = ({
  place,
  placeTotals,
  projectTotal,
  showShared,
  onPlaceChange,
}: WorkflowLibraryPlaceSelectProps) => {
  const { i18n, t } = useTranslation();
  const options = useMemo(() => {
    const numberFormat = new Intl.NumberFormat(i18n.resolvedLanguage);

    return PLACES.filter((item) => item.value !== 'shared' || showShared).map((item): PlaceOption => {
      const total = item.value === 'project' ? projectTotal : placeTotals[item.value];

      return {
        count: total === undefined ? undefined : numberFormat.format(total),
        label: t(item.labelKey),
        value: item.value,
      };
    });
  }, [i18n.resolvedLanguage, placeTotals, projectTotal, showShared, t]);
  const collection = useMemo(() => createListCollection({ items: options }), [options]);
  const value = useMemo(() => [place], [place]);
  const current = options.find((option) => option.value === place);
  const valueText = useMemo(
    () => (current ? <PlaceOptionLabel count={current.count} label={current.label} /> : undefined),
    [current]
  );
  const handleValueChange = useCallback(
    ({ value: [next] }: { value: string[] }) => {
      const option = options.find((candidate) => candidate.value === next);

      if (option) {
        onPlaceChange(option.value);
      }
    },
    [onPlaceChange, options]
  );

  return (
    <Select
      aria-label={t('workflowLibrary.places')}
      collection={collection}
      data-workflow-library-place-select
      flexShrink={0}
      maxW="14rem"
      minW="9rem"
      renderItem={renderPlaceOption}
      value={value}
      valueText={valueText}
      w="auto"
      onValueChange={handleValueChange}
    />
  );
};

const TAG_FILTER_POSITIONING = { placement: 'bottom-start' } as const;

/**
 * The narrow layout's tags button, in the toolbar after the places select: "Tags N" opens the tag checkboxes in a
 * popover. Absent while the place has no tags and none is checked.
 */
export const WorkflowLibraryTagButton = ({
  checkedTags,
  getFallbackFocus,
  tagCounts,
  onTagsChange,
}: WorkflowLibraryTagGroupProps) => {
  const { i18n, t } = useTranslation();
  const triggerRef = useRef<HTMLButtonElement>(null);
  // Inside the popover the trigger outlives every tag; it leaves only with the whole button.
  const getPopoverFallback = useCallback(() => triggerRef.current ?? getFallbackFocus(), [getFallbackFocus]);

  if (tagCounts.length === 0 && checkedTags.length === 0) {
    return null;
  }

  return (
    <Popover.Root lazyMount positioning={TAG_FILTER_POSITIONING} unmountOnExit>
      <Popover.Trigger asChild>
        <Button ref={triggerRef} data-workflow-library-tags-trigger flexShrink={0} gap="1.5" variant="outline">
          {t('workflowLibrary.tags')}
          {checkedTags.length > 0 ? (
            <Text as="span" color="fg.muted" fontVariantNumeric="tabular-nums">
              {new Intl.NumberFormat(i18n.resolvedLanguage).format(checkedTags.length)}
            </Text>
          ) : null}
        </Button>
      </Popover.Trigger>
      <Portal>
        <Popover.Positioner>
          <PopoverContent data-workflow-library-tag-popover p="1.5" w="16rem">
            <Scrollable maxH="min(20rem, calc(var(--available-height) - 1rem))">
              <WorkflowLibraryTagGroup
                checkedTags={checkedTags}
                getFallbackFocus={getPopoverFallback}
                tagCounts={tagCounts}
                onTagsChange={onTagsChange}
              />
            </Scrollable>
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};
