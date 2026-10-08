import { Badge, Box, chakra, HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { AccountMenuSection, useCapabilities, useHasAccountSection } from '@features/identity';
import { getQueueSummary } from '@features/queue/contracts';
import { APP_VERSION, DOCS_URL } from '@platform/runtime/appMetadata';
import { IconButton } from '@platform/ui/Button';
import { InvokeMark } from '@platform/ui/InvokeMark';
import { MenuContent } from '@platform/ui/Menu';
import { Tooltip } from '@platform/ui/Tooltip';
import { DiscordIcon, LightbulbFilamentIcon } from '@platform/ui/VendoredIcon';
import { useQueryClient } from '@tanstack/react-query';
import { useNavigate } from '@tanstack/react-router';
import { OPEN_COMMAND_PALETTE_HOTKEY } from '@workbench/hotkeys/catalog';
import { openCommandPalette } from '@workbench/palette/paletteStore';
import { openWorkbenchSettings } from '@workbench/settings/settingsDialogStore';
import { openWhatsNew } from '@workbench/shell/whatsNewStore';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useActiveProjectId, useActiveProjectSelector } from '@workbench/WorkbenchContext';
import {
  BookOpenTextIcon,
  BlocksIcon,
  BoxIcon,
  ChevronDownIcon,
  FolderIcon,
  HouseIcon,
  ListOrderedIcon,
  SearchIcon,
  SettingsIcon,
  TypeIcon,
} from 'lucide-react';
import { useCallback, useState, type ComponentType, type ElementType } from 'react';
import { useTranslation } from 'react-i18next';

import { useTopbarShortcut } from './useTopbarShortcut';

const MENU_POSITIONING = { placement: 'bottom-start' } as const;
const DISCORD_URL = 'https://discord.gg/ZmtBAhwWhy';
// Loaded on trigger hover or focus, so neither the chunk nor its request is part of route startup.
const loadDonationMenuItem = () => import('@workbench/shell/DonationMenuItem');

export const AppMenu = () => {
  const { t } = useTranslation();
  const { canManageModels, canManageNodes } = useCapabilities();
  const hasAccount = useHasAccountSection();
  const navigate = useNavigate();
  const projectId = useActiveProjectId();
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  const queuedCount = useActiveProjectSelector((project) => getQueueSummary(project.queue.items).total);

  const openHome = useCallback(() => {
    void navigate({ to: '/' });
  }, [navigate]);
  const openProjects = useCallback(() => {
    void navigate({ to: '/projects' });
  }, [navigate]);
  const openModels = useCallback(() => {
    void navigate({ search: { project: projectId }, to: '/models' });
  }, [navigate, projectId]);
  const openNodes = useCallback(() => {
    void navigate({ to: '/nodes' });
  }, [navigate]);
  const openFonts = useCallback(() => {
    void navigate({ to: '/fonts' });
  }, [navigate]);
  const openQueue = useCallback(() => openWorkbenchWidget('queue'), [openWorkbenchWidget]);
  const openSettings = useCallback(() => openWorkbenchSettings(), []);
  const queryClient = useQueryClient();
  // Rendered only once loaded: a lazy() boundary would suspend on open and React delays its reveal, shifting rows.
  // A chunk that fails to load (e.g. after an upgrade) leaves the optional link absent.
  const [DonationMenuItem, setDonationMenuItem] = useState<ComponentType | null>(null);
  const preloadDonationMenuItem = useCallback(() => {
    void loadDonationMenuItem().then(
      (module) => {
        setDonationMenuItem(() => module.DonationMenuItem);
        return module.prefetchDonationMenuItem(queryClient);
      },
      () => undefined
    );
  }, [queryClient]);
  // Assistive technology can activate the trigger with a bare click, without hovering or focusing it first.
  const handleOpenChange = useCallback(
    ({ open }: { open: boolean }) => {
      if (open) {
        preloadDonationMenuItem();
      }
    },
    [preloadDonationMenuItem]
  );

  return (
    <Menu.Root lazyMount positioning={MENU_POSITIONING} onOpenChange={handleOpenChange}>
      <Menu.Trigger asChild>
        <IconButton
          aria-label={t('topbar.appMenu.open')}
          className="group"
          pe="1.5"
          size="lg"
          variant="ghost"
          onFocus={preloadDonationMenuItem}
          onPointerEnter={preloadDonationMenuItem}
        >
          <AppMenuGlyph />
        </IconButton>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="15rem">
            <HStack justify="space-between" px="3" py="2">
              <Text fontWeight="800">Invoke</Text>
              <Text color="fg.subtle" fontSize="xs">
                v{APP_VERSION}
              </Text>
            </HStack>
            <Menu.Separator />
            <Menu.Item value="home" onClick={openHome}>
              <Icon as={HouseIcon} boxSize="3.5" />
              <Menu.ItemText>{t('launchpad.sections.home')}</Menu.ItemText>
            </Menu.Item>
            <Menu.Separator />
            <Menu.ItemGroup>
              <Menu.ItemGroupLabel color="fg.subtle" fontSize="xs" textTransform="uppercase">
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
              <Menu.Item value="fonts" onClick={openFonts}>
                <Icon as={TypeIcon} boxSize="3.5" />
                <Menu.ItemText>{t('launchpad.sections.fonts')}</Menu.ItemText>
              </Menu.Item>
              <Menu.Item value="queue" onClick={openQueue}>
                <Icon as={ListOrderedIcon} boxSize="3.5" />
                <Menu.ItemText>{t('widgets.labels.queue')}</Menu.ItemText>
                {queuedCount > 0 ? (
                  <Badge colorPalette="accent" fontSize="xs" ms="auto" variant="surface">
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
              <AppMenuAction
                icon={LightbulbFilamentIcon}
                label={t('whatsNew.whatsNewInInvoke')}
                value="whats-new"
                onClick={openWhatsNew}
              />
              <AppMenuLink
                href={DOCS_URL}
                icon={BookOpenTextIcon}
                label={t('topbar.appMenu.documentation')}
                value="documentation"
              />
              <AppMenuLink href={DISCORD_URL} icon={DiscordIcon} label="Discord" value="discord" />
            </HStack>
            {DonationMenuItem ? <DonationMenuItem /> : null}
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
  const shortcut = useTopbarShortcut(OPEN_COMMAND_PALETTE_HOTKEY.commandId);
  const label = shortcut ? t('commandPalette.buttonTooltip', { hotkey: shortcut }) : t('commandPalette.buttonLabel');

  return <AppMenuAction icon={SearchIcon} label={label} value="command-palette" onClick={openCommandPalette} />;
};

const SettingsMenuAction = ({ onClick }: { onClick: () => void }) => {
  const { t } = useTranslation();
  // Reads the effective binding, so a remapped or unbound shortcut is never
  // advertised as one that still works.
  const shortcut = useTopbarShortcut('app.openSettings');
  const label = shortcut ? t('common.settingsWithShortcut', { hotkey: shortcut }) : t('common.settings');

  return <AppMenuAction icon={SettingsIcon} label={label} value="settings" onClick={onClick} />;
};

const FOOTER_ITEM_PROPS = {
  alignItems: 'center',
  flex: '0 0 auto',
  h: 'control.md',
  justifyContent: 'center',
  minW: 'control.md',
  p: '0',
  w: 'control.md',
} as const;

const AppMenuAction = ({
  icon,
  label,
  onClick,
  value,
}: {
  /** Lucide for generic actions, a vendored glyph where one is kept for continuity. */
  icon: ElementType;
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
  /** Lucide for generic destinations, a brand mark where the destination has one. */
  icon: ElementType;
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
