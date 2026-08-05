import type { LucideIcon } from 'lucide-react';

import { chakra, HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { APP_VERSION } from '@platform/runtime/appMetadata';
import { Button } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import {
  BookOpenTextIcon,
  ChevronRightIcon,
  ClapperboardIcon,
  CircleQuestionMarkIcon,
  CodeIcon,
  MessagesSquareIcon,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';

/**
 * Docs and community, pinned to the bottom of the Launchpad rail. This used to
 * be two outlined cards — the only card-styled objects on the surface, and
 * visually heavier than the section nav above them. One menu button says the
 * same thing without competing with the navigation for attention.
 */

const MENU_POSITIONING = { placement: 'right-end' } as const;
const GROUP_LABEL_PROPS = { color: 'fg.subtle', fontSize: '2xs', textTransform: 'uppercase' } as const;
const TRIGGER_JUSTIFY = { justifyContent: 'space-between' } as const;

interface HelpLink {
  href: string;
  icon: LucideIcon;
  labelKey: string;
  value: string;
}

const GUIDES: HelpLink[] = [
  {
    href: 'https://invoke-ai.github.io/InvokeAI/',
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
    icon: MessagesSquareIcon,
    labelKey: 'launchpad.help.discord',
    value: 'discord',
  },
  {
    href: 'https://github.com/invoke-ai/InvokeAI',
    icon: CodeIcon,
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

export const HelpMenu = () => {
  const { t } = useTranslation();

  return (
    <Menu.Root positioning={MENU_POSITIONING}>
      <Menu.Trigger asChild>
        <Button
          aria-label={t('launchpad.help.label')}
          color="fg.muted"
          css={TRIGGER_JUSTIFY}
          size="xs"
          variant="ghost"
          w="full"
        >
          <Icon as={CircleQuestionMarkIcon} boxSize="3.5" />
          <Text flex="1" textAlign="start" truncate>
            {t('launchpad.help.label')}
          </Text>
          <Icon as={ChevronRightIcon} boxSize="3" />
        </Button>
      </Menu.Trigger>
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
            </Menu.ItemGroup>
            <Menu.Separator />
            <HStack justify="space-between" px="3" py="1.5">
              <Text fontSize="2xs" fontWeight="700">
                Invoke
              </Text>
              <Text color="fg.subtle" fontSize="2xs">
                {t('launchpad.help.version', { version: APP_VERSION })}
              </Text>
            </HStack>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
