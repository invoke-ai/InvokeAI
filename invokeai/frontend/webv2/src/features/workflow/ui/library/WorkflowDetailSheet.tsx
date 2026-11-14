import type { ProjectGraphState } from '@features/workflow/contracts';

import { Box, HStack, Menu, Portal, Stack, type StackProps, type SystemStyleObject, Text } from '@chakra-ui/react';
import { Button, IconButton } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { ChevronDownIcon, ChevronUpIcon, EllipsisIcon } from 'lucide-react';
import {
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
  type RefObject,
  useCallback,
  useId,
  useMemo,
  useRef,
  useState,
} from 'react';
import { flushSync } from 'react-dom';
import { useTranslation } from 'react-i18next';

import type { WorkflowPreviewStepperProps } from './WorkflowPreviewStepper';

import { keepMenuOpenWhenSiblingMenuCloses } from './WorkflowLibraryCard';
import { WORKFLOW_LIBRARY_NARROW } from './WorkflowLibraryRail';
import { WorkflowThumbnailPlaceholder } from './WorkflowLibraryThumbnail';
import { WorkflowPreviewStepper } from './WorkflowPreviewStepper';
import { WORKFLOW_THUMBNAIL_ASPECT_RATIO, WorkflowThumbnailImage } from './workflowThumbnail';

/**
 * The selected workflow's detail rail, beside the grid in the wide layout and a bottom sheet in the narrow one
 * (`WORKFLOW_LIBRARY_NARROW`). The sheet is a one-row summary (thumbnail, name, facts, Open, More) pinned under the
 * grid; its "Details and requirements" disclosure lays the full rail over the grid, which keeps its scroll position.
 * The layout switch is CSS alone: in the narrow layout the rail's root stops generating a box (`display: contents`),
 * so its details and summary become items of the panes grid (`WORKFLOW_DETAIL_PANES_CSS`). Expanding never animates.
 * Expanded, the sheet is compact: the pinned summary is its header (small thumbnail, the full name, Open), and the
 * details above it drop the rail's large preview and heading (`WORKFLOW_DETAIL_WIDE_ONLY_CSS`), leaving chips,
 * description, requirements and facts in one scrolling column.
 */

const DETAIL_RAIL_WIDTH = '18rem';
/**
 * A content box too short for the panes to share its height: the column scrolls as a whole, the grid or list takes a
 * fixed viewport of its own, and the rail takes a height the dialog body of a 320px-tall window still fits.
 */
export const WORKFLOW_LIBRARY_FLOW_LAYOUT =
  '@container workflow-library-content ((width < 32rem) and (height < 32rem)) or (height < 20rem)';

/** The grid and the detail rail side by side; in the narrow layout, the grid over the sheet's summary. */
export const WORKFLOW_DETAIL_PANES_CSS: SystemStyleObject = {
  [WORKFLOW_LIBRARY_FLOW_LAYOUT]: { flex: '0 0 auto' },
  [WORKFLOW_LIBRARY_NARROW]: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr)',
    gridTemplateRows: 'minmax(0, 1fr) auto',
    rowGap: '2',
    '& > :not([data-detail-sheet])': { gridArea: '1 / 1' },
    // Under the expanded details the grid or list keeps its layout and scroll position, but leaves the tab order. The
    // rule names its target by attribute: a `:not(...)` target makes every :has() check restyle the whole collection,
    // which rows mounting while scrolling trigger each frame.
    '&:has(> [data-detail-sheet][data-expanded]) > [data-workflow-library-collection]': { visibility: 'hidden' },
  },
};

const RAIL_CSS: SystemStyleObject = {
  [WORKFLOW_LIBRARY_FLOW_LAYOUT]: { flex: '0 0 auto', height: '17rem' },
  [WORKFLOW_LIBRARY_NARROW]: { display: 'contents' },
};
const DETAILS_CSS: SystemStyleObject = {
  [WORKFLOW_LIBRARY_NARROW]: {
    bg: 'bg.subtle',
    borderColor: 'border.subtle',
    borderWidth: '1px',
    // Sized by the grid's row alone: in a flow layout the row is the grid's fixed viewport, which the details'
    // content must not stretch.
    contain: 'size',
    display: 'none',
    gridArea: '1 / 1',
    rounded: 'md',
    zIndex: 1,
    '&[data-expanded]': { display: 'flex' },
  },
};
const SUMMARY_CSS: SystemStyleObject = {
  display: 'none',
  [WORKFLOW_LIBRARY_NARROW]: {
    bg: 'bg.subtle',
    borderColor: 'border.subtle',
    borderWidth: '1px',
    bottom: '0',
    display: 'flex',
    flexDirection: 'column',
    gap: '1',
    gridArea: '2 / 1',
    p: '2',
    // A flow layout scrolls the column; the summary and its Open stay in view at its foot.
    position: 'sticky',
    rounded: 'md',
    zIndex: 2,
    // The column's bottom scroll padding keeps focused controls clear of this summary; the summary's own controls
    // are always in view, so focusing them must not scroll the column.
    '& button': { scrollMarginBottom: '-6rem' },
  },
};
const PLACEHOLDER_CSS: SystemStyleObject = { ...RAIL_CSS, [WORKFLOW_LIBRARY_NARROW]: { display: 'none' } };

/**
 * The rail's Preview button: kept mounted in preview mode, which the stage already shows, so leaving can hand focus back
 * to it. The dialog body marks preview mode.
 */
export const WORKFLOW_PREVIEW_TRIGGER_CSS: SystemStyleObject = { '[data-previewing] &': { display: 'none' } };

/** Rail chrome the sheet's summary already carries: the large preview, the heading with Open, and More. */
export const WORKFLOW_DETAIL_WIDE_ONLY_CSS: SystemStyleObject = { [WORKFLOW_LIBRARY_NARROW]: { display: 'none' } };

/** The empty rail while nothing is selected; the narrow layout shows no sheet. */
export const WorkflowDetailRailPlaceholder = () => (
  <Box
    borderColor="border.subtle"
    borderWidth="1px"
    css={PLACEHOLDER_CSS}
    flexShrink={0}
    minH="0"
    rounded="md"
    w={DETAIL_RAIL_WIDTH}
  />
);

/** Names the node packs a workflow needs and this server lacks, beside the requirements. */
export const WorkflowMissingNodePacks = ({ packs }: { packs: readonly string[] }) => {
  const { i18n, t } = useTranslation();
  const listFormat = useMemo(
    () => new Intl.ListFormat(i18n.resolvedLanguage, { type: 'conjunction' }),
    [i18n.resolvedLanguage]
  );

  return packs.length > 0 ? (
    <Text color="fg.error" data-workflow-missing-node-pack-names fontSize="xs" overflowWrap="anywhere">
      {t('workflowLibrary.missingNodePackNames', { count: packs.length, packs: listFormat.format(packs) })}
    </Text>
  ) : null;
};

/** A small 3:2 thumbnail, the image while it loads, else the silhouette: the sheet's summary and the list's rows. */
export const WorkflowSummaryThumbnail = ({
  src,
  w = '16',
  workflowDocument,
}: {
  src: string | null | undefined;
  /** Width; the height follows at 3:2. */
  w?: string;
  workflowDocument: ProjectGraphState | null;
}) => {
  const [failedSrc, setFailedSrc] = useState<string | null>(null);
  const handleError = useCallback(() => setFailedSrc(src ?? null), [src]);

  return (
    <Box
      aria-hidden
      aspectRatio={WORKFLOW_THUMBNAIL_ASPECT_RATIO}
      bg="bg.muted"
      flexShrink={0}
      overflow="hidden"
      rounded="sm"
      w={w}
    >
      {src && src !== failedSrc ? (
        <WorkflowThumbnailImage src={src} onError={handleError} />
      ) : (
        <WorkflowThumbnailPlaceholder workflowDocument={workflowDocument} />
      )}
    </Box>
  );
};

const isRendered = (element: Element | null | undefined): element is HTMLElement =>
  element instanceof HTMLElement && element.getClientRects().length > 0;

/**
 * Whether an Escape keydown belongs to an expanded sheet, which collapses rather than letting the dialog close. Only
 * the narrow layout renders the disclosure, so a sheet left expanded across a widening never swallows Escape.
 */
export const isWorkflowDetailSheetEscape = (event: KeyboardEvent): boolean =>
  event.target instanceof Element &&
  isRendered(event.target.closest('[data-detail-sheet][data-expanded]')?.querySelector('[data-detail-sheet-toggle]'));

export interface WorkflowDetailSheetProps extends Omit<StackProps, 'children' | 'css'> {
  ref: RefObject<HTMLDivElement | null>;
  /** The shown workflow; moving the selection collapses an expanded sheet. */
  workflowId: string;
  /** The full rail: its scroll area and action footer. */
  children: ReactNode;
  name: string;
  /** The summary's facts line, such as "SDXL · 23 nodes · Ready to run"; empty for none. */
  facts: string;
  thumbnailSrc: string | null | undefined;
  workflowDocument: ProjectGraphState | null;
  openLabel: string;
  /** Disables the summary's Open; the facts line is its reason. */
  isOpenDisabled?: boolean;
  onOpen: () => void;
  /** The rail's actions menu items, offered again from the summary's More. */
  actions: ReactNode;
  menuPositioning: Menu.RootProps['positioning'];
  /**
   * Preview mode's stepper, which steps the workflow this rail describes: atop the rail, or beside the name in the
   * narrow sheet's summary.
   */
  stepper?: WorkflowPreviewStepperProps | null;
}

export const WorkflowDetailSheet = ({
  actions,
  children,
  facts,
  isOpenDisabled = false,
  menuPositioning,
  name,
  openLabel,
  ref,
  thumbnailSrc,
  workflowDocument,
  workflowId,
  onOpen,
  stepper,
  ...railProps
}: WorkflowDetailSheetProps) => {
  const { t } = useTranslation();
  const detailsId = useId();
  const factsId = useId();
  const moreActionsIds = useTooltipTriggerIds();
  const detailsRef = useRef<HTMLDivElement>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);
  // Keyed by workflow, so a new selection (a duplicate, a delete, another place) starts collapsed.
  const [expandedWorkflowId, setExpandedWorkflowId] = useState<string | null>(null);
  const isExpanded = expandedWorkflowId === workflowId;
  // Where expanding scrolled the dialog's columns from; collapsing puts them back, so the toolbar returns to view.
  const scrollBeforeExpandRef = useRef<ReadonlyArray<readonly [HTMLElement, number]>>([]);

  const collapse = useCallback(() => {
    flushSync(() => setExpandedWorkflowId(null));

    for (const [element, scrollTop] of scrollBeforeExpandRef.current) {
      element.scrollTop = scrollTop;
    }

    scrollBeforeExpandRef.current = [];
    toggleRef.current?.focus({ preventScroll: true });
  }, []);
  const handleToggle = useCallback(() => {
    if (isExpanded) {
      collapse();
      return;
    }

    const scrolled: [HTMLElement, number][] = [];

    for (let element = detailsRef.current?.parentElement; element; element = element.parentElement) {
      scrolled.push([element, element.scrollTop]);
    }

    scrollBeforeExpandRef.current = scrolled;
    flushSync(() => setExpandedWorkflowId(workflowId));
    // Into the details just shown, so Tab walks them before returning to the summary's actions below. The container
    // takes focus, not its scroll region: Chromium only focuses scrollers that overflow, which the compact details
    // may stop doing as they settle.
    const details = detailsRef.current;

    details?.focus();

    // A flow layout scrolls the column: bring the details' top into view rather than leaving them under the summary.
    details?.scrollIntoView({ block: 'start', inline: 'nearest' });
  }, [collapse, isExpanded, workflowId]);
  const handleKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      // Captured: controls inside the details (tooltip triggers) stop Escape from bubbling. Portalled menus and
      // dialogs reach here through React too, so only keys from the sheet's own DOM collapse it.
      if (
        event.key === 'Escape' &&
        isExpanded &&
        event.target instanceof Node &&
        event.currentTarget.contains(event.target) &&
        isRendered(toggleRef.current)
      ) {
        collapse();
      }
    },
    [collapse, isExpanded]
  );

  return (
    <Stack
      ref={ref}
      borderColor="border.subtle"
      borderWidth="1px"
      css={RAIL_CSS}
      data-detail-sheet
      data-expanded={isExpanded ? '' : undefined}
      flexShrink={0}
      gap="0"
      minH="0"
      rounded="md"
      w={DETAIL_RAIL_WIDTH}
      onKeyDownCapture={handleKeyDown}
      {...railProps}
    >
      <Stack
        ref={detailsRef}
        css={DETAILS_CSS}
        data-detail-sheet-details
        aria-label={name}
        data-expanded={isExpanded ? '' : undefined}
        flex="1"
        gap="0"
        id={detailsId}
        role="group"
        tabIndex={-1}
        minH="0"
        minW="0"
      >
        {stepper ? (
          <Box
            borderBottomWidth="1px"
            borderColor="border.subtle"
            css={WORKFLOW_DETAIL_WIDE_ONLY_CSS}
            data-detail-rail-stepper=""
            flexShrink={0}
            px="2.5"
            py="2.5"
          >
            <WorkflowPreviewStepper {...stepper} />
          </Box>
        ) : null}
        {children}
      </Stack>
      <Box css={SUMMARY_CSS} data-detail-sheet-summary minW="0">
        <HStack gap="2.5" minW="0">
          <WorkflowSummaryThumbnail src={thumbnailSrc} workflowDocument={workflowDocument} />
          <Stack flex="1" gap="0.5" minW="0">
            {/* Expanded, the summary heads the details, so the name reads in full. */}
            <Text fontSize="md" fontWeight="600" minW="0" overflowWrap="anywhere" truncate={!isExpanded}>
              {name}
            </Text>
            {facts ? (
              // One line keeps the summary short enough to stay pinned in a 320px-tall window; the details list
              // every fact.
              <Text color="fg.muted" fontSize="xs" id={factsId} minW="0" truncate>
                {facts}
              </Text>
            ) : null}
          </Stack>
          {stepper ? (
            <Box data-detail-sheet-stepper="">
              <WorkflowPreviewStepper {...stepper} />
            </Box>
          ) : null}
          <Button
            aria-describedby={isOpenDisabled && facts ? factsId : undefined}
            data-detail-sheet-open
            disabled={isOpenDisabled}
            flexShrink={0}
            size="sm"
            onClick={onOpen}
          >
            {openLabel}
          </Button>
          <Menu.Root
            ids={moreActionsIds}
            positioning={menuPositioning}
            onRequestDismiss={keepMenuOpenWhenSiblingMenuCloses}
          >
            {/* Inside a dialog the tooltip must sit inside the menu trigger: wrapped the other way round, the
                menu never takes focus and the first pointer move onto it closes it. */}
            <Menu.Trigger asChild>
              <Tooltip content={t('workflowLibrary.moreActions')} ids={moreActionsIds}>
                <IconButton aria-label={t('workflowLibrary.moreActions')} flexShrink={0} size="sm" variant="outline">
                  <EllipsisIcon />
                </IconButton>
              </Tooltip>
            </Menu.Trigger>
            <Portal>
              <Menu.Positioner>
                <MenuContent minW="min(13rem, var(--available-width))" maxW="var(--available-width)">
                  {actions}
                </MenuContent>
              </Menu.Positioner>
            </Portal>
          </Menu.Root>
        </HStack>
        <Button
          ref={toggleRef}
          alignSelf="flex-start"
          aria-controls={detailsId}
          aria-expanded={isExpanded}
          data-detail-sheet-toggle
          gap="1"
          size="xs"
          variant="plain"
          onClick={handleToggle}
        >
          {isExpanded ? <ChevronDownIcon /> : <ChevronUpIcon />}
          {isExpanded ? t('workflowLibrary.hideDetails') : t('workflowLibrary.detailsAndRequirements')}
        </Button>
      </Box>
    </Stack>
  );
};
