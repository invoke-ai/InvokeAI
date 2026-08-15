import { Badge, Box, chakra, HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { AccountMenuSection, useCapabilities, useHasAccountSection } from '@features/identity';
import { getQueueSummary } from '@features/queue/contracts';
import { APP_VERSION } from '@platform/runtime/appMetadata';
import { IconButton } from '@platform/ui/Button';
import { InvokeMark } from '@platform/ui/InvokeMark';
import { MenuContent } from '@platform/ui/Menu';
import { Tooltip } from '@platform/ui/Tooltip';
import { useNavigate } from '@tanstack/react-router';
import { OPEN_COMMAND_PALETTE_HOTKEY } from '@workbench/hotkeys/catalog';
import { openCommandPalette } from '@workbench/palette/paletteStore';
import { openWorkbenchSettings } from '@workbench/settings/settingsDialogStore';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useActiveProjectId, useActiveProjectSelector } from '@workbench/WorkbenchContext';
import {
  BookOpenTextIcon,
  BlocksIcon,
  BoxIcon,
  ChevronDownIcon,
  FolderIcon,
  ListOrderedIcon,
  MessagesSquareIcon,
  SearchIcon,
  SettingsIcon,
  type LucideIcon,
} from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { useTopbarShortcut } from './useTopbarShortcut';

const MENU_POSITIONING = { placement: 'bottom-start' } as const;
const DOCS_URL = 'https://invoke-ai.github.io/InvokeAI/';
const DISCORD_URL = 'https://discord.gg/ZmtBAhwWhy';

export const AppMenu = () => {
  const { t } = useTranslation();
  const { canManageModels, canManageNodes } = useCapabilities();
  const hasAccount = useHasAccountSection();
  const navigate = useNavigate();
  const projectId = useActiveProjectId();
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  const queuedCount = useActiveProjectSelector((project) => getQueueSummary(project.queue.items).total);

  const openProjects = useCallback(() => {
    void navigate({ to: '/projects' });
  }, [navigate]);
  const openModels = useCallback(() => {
    void navigate({ search: { project: projectId }, to: '/models' });
  }, [navigate, projectId]);
  const openNodes = useCallback(() => {
    void navigate({ to: '/nodes' });
  }, [navigate]);
  const openQueue = useCallback(() => openWorkbenchWidget('queue'), [openWorkbenchWidget]);
  const openSettings = useCallback(() => openWorkbenchSettings(), []);

  return (
    <Menu.Root positioning={MENU_POSITIONING}>
      <Menu.Trigger asChild>
        <IconButton aria-label={t('topbar.appMenu.open')} className="group" pe="1.5" size="xs" variant="ghost">
          <AppMenuGlyph />
        </IconButton>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="15rem">
            <HStack justify="space-between" px="3" py="2">
              <Text fontSize="xs" fontWeight="800">
                Invoke
              </Text>
              <Text color="fg.subtle" fontSize="2xs">
                v{APP_VERSION}
              </Text>
            </HStack>
            <Menu.Separator />
            <Menu.ItemGroup>
              <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                {t('topbar.appMenu.manage')}
              </Menu.ItemGroupLabel>
              <Menu.Item value="projects" onClick={openProjects}>
                <Icon as={FolderIcon} boxSize="3.5" />
                <Menu.ItemText>{t('launchpad.sections.projects')}</Menu.ItemText>
              </Menu.Item>
              {canManageModels ? (
                <Menu.Item value="models" onClick={openModels}>
                  <Icon as={BoxIcon} boxSize="3.5" />
                  <Menu.ItemText>{t('models.manager')}</Menu.ItemText>
                </Menu.Item>
              ) : null}
              {canManageNodes ? (
                <Menu.Item value="nodes" onClick={openNodes}>
                  <Icon as={BlocksIcon} boxSize="3.5" />
                  <Menu.ItemText>{t('nodes.manager')}</Menu.ItemText>
                </Menu.Item>
              ) : null}
              <Menu.Item value="queue" onClick={openQueue}>
                <Icon as={ListOrderedIcon} boxSize="3.5" />
                <Menu.ItemText>{t('widgets.labels.queue')}</Menu.ItemText>
                {queuedCount > 0 ? (
                  <Badge colorPalette="accent" fontSize="2xs" ms="auto" variant="surface">
                    {queuedCount}
                  </Badge>
                ) : null}
              </Menu.Item>
            </Menu.ItemGroup>
            {hasAccount ? (
              <>
                <Menu.Separator />
                <AccountMenuSection />
              </>
            ) : null}
            <Menu.Separator />
            <HStack gap="0.5" px="0" py="0">
              <SearchMenuAction />
              <SettingsMenuAction onClick={openSettings} />
              <AppMenuLink
                href={DOCS_URL}
                icon={BookOpenTextIcon}
                label={t('topbar.appMenu.documentation')}
                value="documentation"
              />
              <AppMenuLink href={DISCORD_URL} icon={MessagesSquareIcon} label="Discord" value="discord" />
            </HStack>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

const CHEVRON_HOVER_PROPS = { color: 'fg' } as const;

const AppMenuGlyph = () => (
  <HStack gap="0" role="presentation">
    <Box alignItems="center" display="flex" justifyContent="center" w="8">
      <InvokeMark size={14} />
    </Box>
    <Icon
      as={ChevronDownIcon}
      boxSize="3"
      color="fg.subtle"
      transition="color var(--wb-motion-duration-fast) ease"
      _groupHover={CHEVRON_HOVER_PROPS}
      _groupExpanded={CHEVRON_HOVER_PROPS}
    />
  </HStack>
);

const SearchMenuAction = () => {
  const { t } = useTranslation();
  // Same formatter as every other shortcut hint in the top bar, so the two
  // hinted actions in this row read alike (and macOS gets ⌘K, not "cmd+k").
  const shortcut = useTopbarShortcut(OPEN_COMMAND_PALETTE_HOTKEY.commandId);
  const label = shortcut ? t('commandPalette.buttonTooltip', { hotkey: shortcut }) : t('commandPalette.buttonLabel');

  return <AppMenuAction icon={SearchIcon} label={label} value="command-palette" onClick={openCommandPalette} />;
};

const SettingsMenuAction = ({ onClick }: { onClick: () => void }) => {
  const { t } = useTranslation();
  // Reads the effective binding, so a remapped or unbound shortcut is never
  // advertised as one that still works.
  const shortcut = useTopbarShortcut('app.openSettings');
  const label = t('common.settings');

  return (
    <AppMenuAction
      icon={SettingsIcon}
      label={shortcut ? `${label} (${shortcut})` : label}
      value="settings"
      onClick={onClick}
    />
  );
};

// Sized like the 2xs icon buttons in widget headers; menu items default to a
// much roomier padding than an icon-only action needs.
const FOOTER_ITEM_PROPS = {
  alignItems: 'center',
  flex: '0 0 auto',
  h: '7',
  justifyContent: 'center',
  minW: '7',
  p: '0',
  w: '7',
} as const;

const AppMenuAction = ({
  icon,
  label,
  onClick,
  value,
}: {
  icon: LucideIcon;
  label: string;
  onClick: () => void;
  value: string;
}) => (
  <Menu.Item {...FOOTER_ITEM_PROPS} aria-label={label} value={value} onClick={onClick}>
    <Tooltip content={label} showArrow>
      <Box alignItems="center" display="flex" h="full" justifyContent="center" w="full">
        <Icon as={icon} boxSize="3.5" />
      </Box>
    </Tooltip>
  </Menu.Item>
);

const AppMenuLink = ({
  href,
  icon,
  label,
  value,
}: {
  href: string;
  icon: LucideIcon;
  label: string;
  value: string;
}) => (
  <Menu.Item {...FOOTER_ITEM_PROPS} aria-label={label} asChild value={value}>
    <chakra.a href={href} rel="noreferrer" target="_blank">
      <Tooltip content={label} showArrow>
        <Box alignItems="center" display="flex" h="full" justifyContent="center" w="full">
          <Icon as={icon} boxSize="3.5" />
        </Box>
      </Tooltip>
    </chakra.a>
  </Menu.Item>
);
