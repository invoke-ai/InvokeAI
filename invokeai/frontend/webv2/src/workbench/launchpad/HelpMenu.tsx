import { chakra, HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { APP_VERSION, DOCS_URL } from '@platform/runtime/appMetadata';
import { Button, IconButton } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { DiscordIcon, GithubIcon } from '@platform/ui/VendoredIcon';
import { useQueryClient } from '@tanstack/react-query';
import { BookOpenTextIcon, ChevronRightIcon, ClapperboardIcon, CircleQuestionMarkIcon } from 'lucide-react';
import { useCallback, useState, type ComponentType, type ElementType } from 'react';
import { useTranslation } from 'react-i18next';

const MENU_POSITIONING = { placement: 'right-end' } as const;
/** fg.muted: fg.subtle falls below 4.5:1 on the menu surface at this size. */
const GROUP_LABEL_PROPS = { color: 'fg.muted', fontSize: 'xs', textTransform: 'uppercase' } as const;
const TRIGGER_JUSTIFY = { justifyContent: 'space-between' } as const;
// Loaded on trigger hover or focus, so neither the chunk nor its request is part of route startup.
const loadDonationMenuItem = () => import('@workbench/shell/DonationMenuItem');

interface HelpLink {
  href: string;
  /** Lucide for generic destinations, a brand mark where the destination has one. */
  icon: ElementType;
  labelKey: string;
  value: string;
}

const GUIDES: HelpLink[] = [
  {
    href: DOCS_URL,
    icon: BookOpenTextIcon,
    labelKey: 'launchpad.help.documentation',
    value: 'documentation',
  },
  {
    href: 'https://www.youtube.com/@invokeai',
    icon: ClapperboardIcon,
    labelKey: 'launchpad.help.youtube',
    value: 'youtube',
  },
];

const COMMUNITY: HelpLink[] = [
  {
    href: 'https://discord.gg/ZmtBAhwWhy',
    icon: DiscordIcon,
    labelKey: 'launchpad.help.discord',
    value: 'discord',
  },
  {
    href: 'https://github.com/invoke-ai/InvokeAI',
    icon: GithubIcon,
    labelKey: 'launchpad.help.github',
    value: 'github',
  },
];

const HelpMenuLink = ({ href, icon, labelKey, value }: HelpLink) => {
  const { t } = useTranslation();

  return (
    <Menu.Item asChild value={value}>
      <chakra.a href={href} rel="noreferrer" target="_blank">
        <Icon as={icon} boxSize="3.5" />
        <Menu.ItemText>{t(labelKey)}</Menu.ItemText>
      </chakra.a>
    </Menu.Item>
  );
};

/** `compact` is the icon-only rail: the trigger is named by a tooltip instead of its label. */
export const HelpMenu = ({ compact = false }: { compact?: boolean }) => {
  const { t } = useTranslation();
  const ids = useTooltipTriggerIds();
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
    <Menu.Root ids={ids} lazyMount positioning={MENU_POSITIONING} onOpenChange={handleOpenChange}>
      {compact ? (
        <Tooltip content={t('launchpad.help.label')} ids={ids} placement="right">
          <Menu.Trigger asChild>
            <IconButton
              aria-label={t('launchpad.help.label')}
              color="fg.muted"
              size="lg"
              variant="ghost"
              onFocus={preloadDonationMenuItem}
              onPointerEnter={preloadDonationMenuItem}
            >
              <Icon as={CircleQuestionMarkIcon} boxSize="3.5" />
            </IconButton>
          </Menu.Trigger>
        </Tooltip>
      ) : (
        <Menu.Trigger asChild>
          <Button
            aria-label={t('launchpad.help.label')}
            color="fg.muted"
            css={TRIGGER_JUSTIFY}
            variant="ghost"
            w="full"
            onFocus={preloadDonationMenuItem}
            onPointerEnter={preloadDonationMenuItem}
          >
            <Icon as={CircleQuestionMarkIcon} boxSize="3.5" />
            <Text flex="1" textAlign="start" truncate>
              {t('launchpad.help.label')}
            </Text>
            <Icon as={ChevronRightIcon} boxSize="3" />
          </Button>
        </Menu.Trigger>
      )}
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="13rem">
            <Menu.ItemGroup>
              <Menu.ItemGroupLabel {...GROUP_LABEL_PROPS}>{t('launchpad.help.guides')}</Menu.ItemGroupLabel>
              {GUIDES.map((link) => (
                <HelpMenuLink key={link.value} {...link} />
              ))}
            </Menu.ItemGroup>
            <Menu.Separator />
            <Menu.ItemGroup>
              <Menu.ItemGroupLabel {...GROUP_LABEL_PROPS}>{t('launchpad.help.community')}</Menu.ItemGroupLabel>
              {COMMUNITY.map((link) => (
                <HelpMenuLink key={link.value} {...link} />
              ))}
              {DonationMenuItem ? <DonationMenuItem /> : null}
            </Menu.ItemGroup>
            <Menu.Separator />
            <HStack justify="space-between" px="3" py="1.5">
              <Text fontSize="xs" fontWeight="700">
                Invoke
              </Text>
              <Text color="fg.muted" fontSize="xs">
                {t('launchpad.help.version', { version: APP_VERSION })}
              </Text>
            </HStack>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
