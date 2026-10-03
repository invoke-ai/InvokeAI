import type { SystemStyleObject } from '@chakra-ui/react';
import type { LucideIcon } from 'lucide-react';

import { Icon, Stack, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { LightbulbFilamentIcon } from '@platform/ui/VendoredIcon';
import { Link } from '@tanstack/react-router';
import { openWhatsNew } from '@workbench/shell/whatsNewStore';
import { useId, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { HelpMenu } from './HelpMenu';
import { OpenProjectsNavSection } from './OpenProjectsControl';

const NAV_BORDER_END_WIDTH = { md: '1px' } as const;
const NAV_BORDER_BOTTOM_WIDTH = { base: '1px', md: '0' } as const;
const NAV_WIDTH = { base: 'full', md: '56' } as const;
/** Below md, size the stacked rail to content; full height would consume the fixed shell and leave no page area. */
const NAV_HEIGHT = { base: 'auto', md: 'full' } as const;
const FOOTER_MARGIN_TOP = { md: 'auto' } as const;

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

/** Rail entries match the settings dialog's navigation items: ghost buttons, subtle for the current page. */
const NAV_ITEM_PROPS = { justifyContent: 'start', size: 'lg', w: 'full' } as const;

const NavLink = ({ isActive, item }: { isActive: boolean; item: LaunchpadNavItem }) => (
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
const WhatsNewButton = () => {
  const { t } = useTranslation();

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
  group,
  items,
  showLabel,
}: {
  activeId: string;
  group: Exclude<LaunchpadNavGroupId, 'footer'>;
  items: LaunchpadNavItem[];
  showLabel: boolean;
}) => {
  const { t } = useTranslation();
  const headingId = useId();

  return (
    <Stack aria-labelledby={showLabel ? headingId : undefined} as="section" gap="0.5">
      {showLabel ? (
        <Text css={GROUP_LABEL_SX} id={headingId}>
          {t(GROUP_LABEL_KEY[group])}
        </Text>
      ) : null}
      {items.map((item) => (
        <NavLink key={item.id} isActive={item.id === activeId} item={item} />
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

  return (
    <Stack
      aria-label={t('launchpad.sectionsLabel')}
      as="nav"
      borderColor="border.subtle"
      borderBottomWidth={NAV_BORDER_BOTTOM_WIDTH}
      borderEndWidth={NAV_BORDER_END_WIDTH}
      css={NAV_SX}
      flexShrink={0}
      gap="0"
      h={NAV_HEIGHT}
      minH="0"
      p="2"
      w={NAV_WIDTH}
    >
      {items.length > 1 ? (
        <NavGroup activeId={activeId} group="workspace" items={workspaceItems} showLabel={showGroupLabels} />
      ) : null}
      <OpenProjectsNavSection headingCss={GROUP_LABEL_SX} itemProps={NAV_ITEM_PROPS} />
      {items.length > 1 && manageItems.length > 0 ? (
        <NavGroup activeId={activeId} group="manage" items={manageItems} showLabel={showGroupLabels} />
      ) : null}

      <Stack gap="0.5" mt={FOOTER_MARGIN_TOP} pt="2">
        {footerItems.map((item) => (
          <NavLink key={item.id} isActive={item.id === activeId} item={item} />
        ))}
        <WhatsNewButton />
        <HelpMenu />
      </Stack>
    </Stack>
  );
};
