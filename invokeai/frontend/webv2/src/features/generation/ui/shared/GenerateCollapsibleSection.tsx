/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop -- one menu per section;
   its content and items mount only while it is open (lazyMount, unmountOnExit). */
import type { KeyboardEvent, MouseEvent, ReactNode } from 'react';

import { Box, Collapsible, Flex, Icon, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { canCopyText, copyText } from '@platform/browser/clipboard';
import { MenuContent, MenuItem, useContextMenu } from '@platform/ui/Menu';
import { ChevronRightIcon, CircleAlertIcon } from 'lucide-react';
import { useCallback, useId, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useGenerateSectionIssueCount } from './GenerateIssues';
import {
  getPanelSections,
  isPanelSectionOpen,
  registerPanelSection,
  soloPanelSection,
  toggleAllPanelSections,
  useSectionPanelId,
} from './sectionPanel';

type Props = {
  label: string;
  defaultOpen?: boolean;
  /** When set, the open state persists per user across reloads under this id. */
  sectionId?: string;
  /**
   * What the fields are set to (a size, a seed, an override count), shown after the title only while the section is
   * folded, as muted text joined by middle dots. Falsy items are skipped.
   */
  summary?: readonly ReactNode[];
  /** Badges that hold whether the section is open or not (an incompatibility, a shared setting). */
  status?: ReactNode;
  /** The fields differ from the defaults `onReset` restores; marks the title with a dot. */
  isChanged?: boolean;
  /** Restores the section's fields to their defaults, offered from the title's context menu. */
  onReset?: () => void;
  /** Names what `onReset` restores when it is less than the whole section; "Reset section to defaults" otherwise. */
  resetLabel?: string;
  children: ReactNode;
};

/**
 * A section reads as a segment tab over its fields: the title is a chip with the `SegmentTabs` values (filled while
 * its section shows, muted while it is folded). The fields sit bare on the panel at the chip's inset, so the bordered
 * fields never nest in a second box, and a full-bleed hairline starts each section. Hairlines mean "section" only;
 * groups inside one use spacing and labels.
 */
const SECTION_STYLES = {
  '&:last-child': { borderBottomWidth: '1px' },
  // Anything brought into view inside the fields (a focused control, a revealed issue) stops below the pinned title
  // rather than under it: the title's height plus a little air.
  '& > [data-part="content"] *': { scrollMarginTop: 'calc(var(--chakra-sizes-7) + var(--chakra-spacing-7))' },
} as const;
const CHIP_SHOWN_BG = 'gray.hoverTint/15';
const COLLAPSIBLE_INDICATOR_OPEN_STYLES = { transform: 'rotate(90deg)' };
/**
 * The fields grow in by a fixed few pixels while the content opens, and back as it folds, on the recipe's duration
 * (1ms under Reduce motion). On the fields, not the content: the content clips its animating height, and Chromium
 * leaves fields unpainted when one element does both that and a composited transform animation.
 */
const FIELDS_STYLES = {
  transformOrigin: 'top',
  '[data-part="content"][data-state="open"] > &': {
    animationDuration: 'moderate',
    animationName: 'wb-section-scale-in',
  },
  // Not held at its end: the content has faded out by then, and a held transform would stay on the folded fields.
  '[data-part="content"][data-state="closed"] > &': {
    animationDuration: 'moderate',
    animationName: 'wb-section-scale-out',
  },
} as const;
const TRIGGER_STYLES = {
  '&[data-state="open"]': { color: 'fg' },
  '&[data-state="open"] > [data-section-chip]': { bg: CHIP_SHOWN_BG },
  '&:not([data-state="open"]):hover': { color: 'fg' },
  '&:not([data-state="open"]):hover > [data-section-chip]': { bg: 'bg.hover' },
  // A section with issues says so on its chip, open or folded.
  '& > [data-section-chip][data-issues]': { bg: 'bg.error', color: 'fg.error' },
  '&:hover > [data-section-chip][data-issues]': { bg: 'red.hoverTint/20' },
  // The row is full-bleed, so the scroller would clip a ring drawn around it; the chip carries the ring, inset.
  '&:focus-visible': { outline: 'none' },
  '&:focus-visible > [data-section-chip]': {
    outline: '2px solid',
    outlineColor: 'accent.solid',
    outlineOffset: '-2px',
  },
  // Summaries say what a folded section holds; an open one shows its fields instead.
  '&[data-state="open"] [data-section-summary]': { opacity: 0, visibility: 'hidden' },
  // A hairline under the title only while it is pinned over the fields scrolling beneath it. Chromium resolves the
  // scroll-state query; elsewhere the title pins without the edge.
  containerType: 'scroll-state',
  '&::after': {
    bg: 'border',
    bottom: 0,
    content: '""',
    h: '1px',
    insetInline: 0,
    opacity: 0,
    pointerEvents: 'none',
    position: 'absolute',
    transition: 'opacity var(--wb-motion-duration-fast)',
  },
  '@container scroll-state(stuck: top)': { '&::after': { opacity: 1 } },
} as const;
// Separators are drawn, so the summary reads as its values; the empty alternative keeps them out of the name.
const SUMMARY_STYLES = { '& > * + *::before': { content: '"·" / ""', marginInline: '1.5' } } as const;

const isShownSummary = (item: ReactNode) => item !== null && item !== undefined && item !== false && item !== '';

const getSummaryText = (root: HTMLElement): string =>
  [...root.querySelectorAll<HTMLElement>('[data-section-summary] > *')]
    .map((item) => item.textContent?.trim() ?? '')
    .filter(Boolean)
    .join(' · ');

interface MenuFacts {
  anyFolded: boolean;
  othersOpen: boolean;
  root: HTMLElement;
}

export const GenerateCollapsibleSection = ({
  children,
  defaultOpen,
  isChanged,
  label,
  onReset,
  resetLabel,
  sectionId,
  status,
  summary,
}: Props) => {
  const { t } = useTranslation();
  const { sectionPreferences } = useGenerationUi();
  const key = useId();
  // Outside a panel (a section rendered on its own) the section is its own panel.
  const panelId = useSectionPanelId() ?? key;
  const [localOpen, setLocalOpen] = useState(defaultOpen ?? false);
  const isOpen =
    sectionId === undefined ? localOpen : (sectionPreferences.sectionsOpen[sectionId] ?? defaultOpen ?? false);
  const issueCount = useGenerateSectionIssueCount(sectionId);
  const { anchor, close, onRequestDismiss, open } = useContextMenu();
  const [menuFacts, setMenuFacts] = useState<MenuFacts | null>(null);
  const shownSummary = summary?.filter(isShownSummary) ?? [];
  const persist = sectionPreferences.setSectionsOpen;

  const setOpen = (nextOpen: boolean) => {
    if (sectionId === undefined) {
      setLocalOpen(nextOpen);
    } else {
      persist({ [sectionId]: nextOpen });
    }
  };

  // Joins the panel while mounted; a new label or id re-registers it.
  const registerRoot = useCallback(
    (root: HTMLDivElement | null) => {
      const trigger = root?.querySelector<HTMLElement>(':scope > [data-part="trigger"]');

      return root && trigger
        ? registerPanelSection(panelId, { key, label, root, sectionId, setLocalOpen, trigger })
        : undefined;
    },
    [key, label, panelId, sectionId]
  );

  const panelSections = () => getPanelSections(panelId);

  // Alt-click solos the section; Shift-click opens every section in the panel, or folds them all when none is folded.
  const handleClick = (event: MouseEvent<HTMLButtonElement>) => {
    if (!(event.altKey || event.shiftKey)) {
      return;
    }

    event.preventDefault();
    const root = event.currentTarget.parentElement!;

    if (event.altKey) {
      soloPanelSection(panelSections(), root, persist);
    } else {
      toggleAllPanelSections(panelSections(), persist);
    }
  };

  // The titles behave as a tree's rows: up and down move between them, right opens, left folds.
  const handleKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) {
      return;
    }

    const triggers = panelSections().map((entry) => entry.trigger);
    const index = triggers.indexOf(event.currentTarget);
    const target =
      event.key === 'ArrowDown'
        ? triggers[index + 1]
        : event.key === 'ArrowUp'
          ? triggers[index - 1]
          : event.key === 'Home'
            ? triggers[0]
            : event.key === 'End'
              ? triggers.at(-1)
              : undefined;

    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      if (isOpen !== (event.key === 'ArrowRight')) {
        setOpen(!isOpen);
      }
    } else if (index !== -1 && ['ArrowDown', 'ArrowUp', 'End', 'Home'].includes(event.key)) {
      // At either end the key stays here rather than scrolling the panel.
      target?.focus();
    } else {
      return;
    }

    event.preventDefault();
  };

  const handleContextMenu = (event: MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    const trigger = event.currentTarget;
    const root = trigger.parentElement!;
    const sections = panelSections();
    // A keyboard-opened menu has no pointer position of its own (see `GenerateFieldContextMenu`).
    const native = event.nativeEvent as Partial<PointerEvent>;
    const fromKeyboard =
      event.button === -1 || native.pointerType === '' || (event.clientX === 0 && event.clientY === 0);
    const rect = trigger.getBoundingClientRect();

    setMenuFacts({
      anyFolded: sections.some((entry) => !isPanelSectionOpen(entry)),
      othersOpen: sections.some((entry) => entry.root !== root && isPanelSectionOpen(entry)),
      root,
    });
    open({
      // A frame late: Zag focuses the menu a frame after a highlight, and a pick within that frame would otherwise
      // land focus back in the closing menu.
      restoreFocus: () => requestAnimationFrame(() => trigger.focus()),
      x: fromKeyboard ? rect.left : event.clientX,
      y: fromKeyboard ? rect.bottom : event.clientY,
    });
  };

  const copyUnavailableReason = canCopyText() ? undefined : t('common.copyUnavailable');

  return (
    <Collapsible.Root
      ref={registerRoot}
      className="generate-section"
      css={SECTION_STYLES}
      data-generate-section={sectionId}
      borderTopWidth="1px"
      open={isOpen}
      onOpenChange={({ open: nextOpen }) => setOpen(nextOpen)}
    >
      {/* Pinned while its section scrolls beneath it, so the fields in view always have their title. */}
      <Collapsible.Trigger
        bg="bg.subtle"
        css={TRIGGER_STYLES}
        display="flex"
        gap={2}
        w="full"
        // The chip sits the same distance from the hairlines above and below (when folded) as from the panel's edge.
        pb="1.5"
        ps="1.5"
        pe={2}
        pt="1.5"
        alignItems="center"
        color="fg.muted"
        position="sticky"
        top={0}
        zIndex={1}
        onClick={handleClick}
        onContextMenu={handleContextMenu}
        onKeyDown={handleKeyDown}
      >
        <Text
          as="span"
          data-issues={issueCount > 0 ? '' : undefined}
          data-section-chip=""
          display="inline-flex"
          alignItems="center"
          flexShrink={0}
          fontSize="md"
          fontWeight="medium"
          gap="1"
          h="7"
          maxW="full"
          minW="0"
          // Less on the chevron's side: its glyph is narrower than its box.
          pe="2.5"
          ps="1.5"
          rounded="control"
          transition="background var(--wb-motion-duration-fast), color var(--wb-motion-duration-fast)"
        >
          <Collapsible.Indicator
            _open={COLLAPSIBLE_INDICATOR_OPEN_STYLES}
            flexShrink={0}
            transition="transform var(--wb-motion-duration-slow)"
          >
            <ChevronRightIcon size="14" />
          </Collapsible.Indicator>
          <Text as="span" truncate>
            {label}
          </Text>
          {isChanged ? (
            <Box as="span" bg="accent.solid" boxSize="1.5" data-section-changed="" flexShrink={0} rounded="full">
              <Text as="span" srOnly>
                {t('widgets.generate.sectionChanged')}
              </Text>
            </Box>
          ) : null}
          {/* A closed section never hides why Invoke is blocked. */}
          {issueCount > 0 ? (
            <Text
              as="span"
              alignItems="center"
              data-section-issues=""
              display="inline-flex"
              flexShrink={0}
              fontVariantNumeric="tabular-nums"
              gap="0.5"
              ms="0.5"
            >
              <Icon as={CircleAlertIcon} aria-hidden boxSize="3" />
              <span aria-hidden>{issueCount}</span>
              <Text as="span" srOnly>
                {t('widgets.generate.issueCount', { count: issueCount })}
              </Text>
            </Text>
          ) : null}
        </Text>

        <Flex alignItems="center" gap={2} minW="0" ml="auto">
          {shownSummary.length > 0 ? (
            <Text
              as="span"
              aria-hidden={isOpen || undefined}
              color="fg.muted"
              css={SUMMARY_STYLES}
              data-section-summary=""
              fontSize="sm"
              fontVariantNumeric="tabular-nums"
              minW="0"
              transition="opacity var(--wb-motion-duration-fast), visibility var(--wb-motion-duration-fast)"
              truncate
            >
              {shownSummary.map((item, index) => (
                // oxlint-disable-next-line react/no-array-index-key -- positional values with no identity of their own.
                <span key={index}>{item}</span>
              ))}
            </Text>
          ) : null}
          {status ? (
            <Flex flexShrink={0} fontVariantNumeric="tabular-nums" gap={1}>
              {status}
            </Flex>
          ) : null}
        </Flex>
      </Collapsible.Trigger>

      {/* Isolated, so a positioned field (a segment thumb, a textarea handle) never paints over the pinned title. */}
      <Collapsible.Content isolation="isolate">
        {/* The section owns its inset and rhythm, so every widget's sections space their fields alike. */}
        <Stack css={FIELDS_STYLES} gap="2.5" px={2} pb={3}>
          {children}
        </Stack>
      </Collapsible.Content>

      <Menu.Root
        lazyMount
        open={anchor !== null}
        positioning={{
          getAnchorRect: () => (anchor ? { height: 1, width: 1, x: anchor.x, y: anchor.y } : null),
          placement: 'bottom-start',
        }}
        onOpenChange={(event) => {
          if (!event.open) {
            close();
          }
        }}
        onRequestDismiss={onRequestDismiss}
        unmountOnExit
      >
        <Portal>
          <Menu.Positioner>
            <MenuContent>
              <MenuItem
                disabled={!menuFacts?.othersOpen}
                value="fold-others"
                onSelect={() => {
                  if (menuFacts) {
                    soloPanelSection(panelSections(), menuFacts.root, persist);
                  }
                }}
              >
                {t('widgets.generate.sectionMenu.foldOthers')}
              </MenuItem>
              <MenuItem value="toggle-all" onSelect={() => toggleAllPanelSections(panelSections(), persist)}>
                {menuFacts?.anyFolded
                  ? t('widgets.generate.sectionMenu.expandAll')
                  : t('widgets.generate.sectionMenu.foldAll')}
              </MenuItem>
              {shownSummary.length > 0 ? (
                <MenuItem
                  unavailableReason={copyUnavailableReason}
                  value="copy-summary"
                  onSelect={() => {
                    if (menuFacts) {
                      void copyText(
                        t('widgets.generate.sectionMenu.summaryText', {
                          label,
                          summary: getSummaryText(menuFacts.root),
                        })
                      ).catch(() => undefined);
                    }
                  }}
                >
                  {t('widgets.generate.sectionMenu.copySummary')}
                </MenuItem>
              ) : null}
              {onReset ? (
                <MenuItem disabled={isChanged === false} value="reset" onSelect={onReset}>
                  {resetLabel ?? t('widgets.generate.sectionMenu.reset')}
                </MenuItem>
              ) : null}
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
    </Collapsible.Root>
  );
};
