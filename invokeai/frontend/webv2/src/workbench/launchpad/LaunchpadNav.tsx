import type { SystemStyleObject } from '@chakra-ui/react';
import type { LucideIcon } from 'lucide-react';

import { Icon, Separator, Stack, Text, VisuallyHidden } from '@chakra-ui/react';
import { Button, IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { LightbulbFilamentIcon } from '@platform/ui/VendoredIcon';
import { Link } from '@tanstack/react-router';
import { system } from '@theme/system';
import { openWhatsNew } from '@workbench/shell/whatsNewStore';
import { useId, useMemo, useSyncExternalStore } from 'react';
import { useTranslation } from 'react-i18next';

import { HelpMenu } from './HelpMenu';
import { OpenProjectsNavSection } from './OpenProjectsControl';

/**
 * Below the theme's md breakpoint the rail keeps its place beside the page but shows icons only, each named by a
 * tooltip: stacked above the page it took most of a short or zoomed window and left the page no height.
 */
const COMPACT_QUERY = system.breakpoints.down('md').replace(/^@media\s+/, '');
const subscribeCompact = (onChange: () => void): (() => void) => {
  const query = globalThis.matchMedia(COMPACT_QUERY);
  query.addEventListener('change', onChange);
  return () => query.removeEventListener('change', onChange);
};
const getCompact = (): boolean => globalThis.matchMedia(COMPACT_QUERY).matches;
const TOOLTIP_PLACEMENT = 'right';

/**
 * Sections are routes, so the rail is route links in visual order: Workspace, the open projects, Manage, then the
 * footer (Preferences, then the What's New and Help actions). Tab and reading order follow what the rail shows.
 */

export type LaunchpadNavGroupId = 'workspace' | 'manage' | 'footer';

export interface LaunchpadNavItem {
  id: string;
  label: string;
  icon: LucideIcon;
  group: LaunchpadNavGroupId;
  to: string;
}

const GROUP_LABEL_KEY: Record<Exclude<LaunchpadNavGroupId, 'footer'>, string> = {
  manage: 'launchpad.groups.manage',
  workspace: 'launchpad.groups.workspace',
};

/** Use fg.muted for rail-heading contrast; the menu label's fg.subtle falls below 4.5:1 here. */
const GROUP_LABEL_SX: SystemStyleObject = {
  color: 'fg.muted',
  fontSize: 'xs',
  fontWeight: '600',
  letterSpacing: '0.02em',
  pb: '1',
  pt: '1',
  px: '2',
  textTransform: 'uppercase',
};
/** Space between rail groups; headings carry no top padding of their own. */
const NAV_SX: SystemStyleObject = { '& > section ~ section': { pt: '2' } };
/** Without headings, a hairline separates the icon groups. */
const COMPACT_NAV_SX: SystemStyleObject = {
  '& > section ~ section': { borderTopWidth: '1px', mt: '1.5', pt: '1.5' },
};

/** Rail entries match the settings dialog's navigation items: ghost buttons, subtle for the current page. */
const NAV_ITEM_PROPS = { justifyContent: 'start', size: 'lg', w: 'full' } as const;

const NavLink = ({ compact, isActive, item }: { compact: boolean; isActive: boolean; item: LaunchpadNavItem }) =>
  compact ? (
    <Tooltip content={item.label} placement={TOOLTIP_PLACEMENT}>
      <IconButton
        asChild
        aria-current={isActive ? 'page' : undefined}
        aria-label={item.label}
        size="lg"
        variant={isActive ? 'subtle' : 'ghost'}
      >
        <Link to={item.to}>
          <Icon as={item.icon} boxSize="3.5" />
        </Link>
      </IconButton>
    </Tooltip>
  ) : (
    <Button
      asChild
      {...NAV_ITEM_PROPS}
      aria-current={isActive ? 'page' : undefined}
      variant={isActive ? 'subtle' : 'ghost'}
    >
      <Link to={item.to}>
        <Icon as={item.icon} boxSize="3.5" flexShrink={0} />
        <Text truncate>{item.label}</Text>
      </Link>
    </Button>
  );

const WHATS_NEW_JUSTIFY = { justifyContent: 'start' } as const;

/** An action, not a route: styled like the Help trigger beneath it rather than as a rail link. */
const WhatsNewButton = ({ compact }: { compact: boolean }) => {
  const { t } = useTranslation();

  if (compact) {
    return (
      <Tooltip content={t('whatsNew.whatsNewInInvoke')} placement={TOOLTIP_PLACEMENT}>
        <IconButton
          aria-label={t('whatsNew.whatsNewInInvoke')}
          color="fg.muted"
          size="lg"
          variant="ghost"
          onClick={openWhatsNew}
        >
          <Icon as={LightbulbFilamentIcon} boxSize="3.5" />
        </IconButton>
      </Tooltip>
    );
  }

  return (
    <Button color="fg.muted" css={WHATS_NEW_JUSTIFY} variant="ghost" w="full" onClick={openWhatsNew}>
      <Icon as={LightbulbFilamentIcon} boxSize="3.5" />
      <Text flex="1" textAlign="start" truncate>
        {t('whatsNew.whatsNewInInvoke')}
      </Text>
    </Button>
  );
};

const NavGroup = ({
  activeId,
  compact,
  group,
  items,
  showLabel,
}: {
  activeId: string;
  compact: boolean;
  group: Exclude<LaunchpadNavGroupId, 'footer'>;
  items: LaunchpadNavItem[];
  showLabel: boolean;
}) => {
  const { t } = useTranslation();
  const headingId = useId();

  return (
    <Stack aria-labelledby={showLabel ? headingId : undefined} as="section" gap="0.5">
      {showLabel ? (
        compact ? (
          <VisuallyHidden id={headingId}>{t(GROUP_LABEL_KEY[group])}</VisuallyHidden>
        ) : (
          <Text css={GROUP_LABEL_SX} id={headingId}>
            {t(GROUP_LABEL_KEY[group])}
          </Text>
        )
      ) : null}
      {items.map((item) => (
        <NavLink key={item.id} compact={compact} isActive={item.id === activeId} item={item} />
      ))}
    </Stack>
  );
};

export const LaunchpadNav = ({ activeId, items }: { activeId: string; items: LaunchpadNavItem[] }) => {
  const { t } = useTranslation();
  const workspaceItems = useMemo(() => items.filter((item) => item.group === 'workspace'), [items]);
  const manageItems = useMemo(() => items.filter((item) => item.group === 'manage'), [items]);
  const footerItems = useMemo(() => items.filter((item) => item.group === 'footer'), [items]);
  // Show headings only when multiple groups need distinguishing.
  const showGroupLabels = workspaceItems.length > 0 && manageItems.length > 0;
  const compact = useSyncExternalStore(subscribeCompact, getCompact);

  return (
    <Stack
      aria-label={t('launchpad.sectionsLabel')}
      as="nav"
      borderColor="border.subtle"
      borderEndWidth="1px"
      css={compact ? COMPACT_NAV_SX : NAV_SX}
      flexShrink={0}
      gap="0"
      h="full"
      minH="0"
      overflowY="auto"
      p={compact ? '1.5' : '2'}
      w={compact ? undefined : '56'}
    >
      {items.length > 1 ? (
        <NavGroup
          activeId={activeId}
          compact={compact}
          group="workspace"
          items={workspaceItems}
          showLabel={showGroupLabels}
        />
      ) : null}
      <OpenProjectsNavSection compact={compact} headingCss={GROUP_LABEL_SX} itemProps={NAV_ITEM_PROPS} />
      {items.length > 1 && manageItems.length > 0 ? (
        <NavGroup
          activeId={activeId}
          compact={compact}
          group="manage"
          items={manageItems}
          showLabel={showGroupLabels}
        />
      ) : null}

      <Stack gap="0.5" mt="auto" pt="2">
        {compact ? <Separator mb="1.5" /> : null}
        {footerItems.map((item) => (
          <NavLink key={item.id} compact={compact} isActive={item.id === activeId} item={item} />
        ))}
        <WhatsNewButton compact={compact} />
        <HelpMenu compact={compact} />
      </Stack>
    </Stack>
  );
};
