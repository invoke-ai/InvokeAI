import type { LaunchpadStartEntry } from '@workbench/launchpad/startEntries';

import { Icon, Menu, Portal } from '@chakra-ui/react';
import { Button, IconButton } from '@platform/ui/Button';
import { Group } from '@platform/ui/Group';
import { MenuActionItem, MenuContent } from '@platform/ui/Menu';
import { Link, useNavigate } from '@tanstack/react-router';
import { LAUNCHPAD_START_ENTRIES } from '@workbench/launchpad/startEntries';
import { ChevronDownIcon, PlusIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

/** Plain clicks use the last arrangement; the caret offers the same starts as the Home tiles. */

const NEW_PROJECT_SEARCH = { new: true } as const;
const MENU_POSITIONING = { placement: 'bottom-end' } as const;

export const NewProjectButton = ({ variant = 'solid' }: { variant?: 'outline' | 'solid' }) => {
  const { t } = useTranslation();

  return (
    <Group attached>
      <Button asChild size="xs" variant={variant}>
        <Link search={NEW_PROJECT_SEARCH} to="/app">
          <Icon as={PlusIcon} boxSize="3.5" />
          {t('projects.newProject')}
        </Link>
      </Button>
      <Menu.Root positioning={MENU_POSITIONING}>
        <Menu.Trigger asChild>
          <IconButton aria-label={t('projects.newProjectStart')} size="xs" variant={variant}>
            <Icon as={ChevronDownIcon} boxSize="3.5" />
          </IconButton>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="16rem">
              <Menu.ItemGroup>
                <Menu.ItemGroupLabel>{t('launchpad.home.intents.heading')}</Menu.ItemGroupLabel>
                {LAUNCHPAD_START_ENTRIES.map((entry) => (
                  <NewProjectStartMenuItem entry={entry} key={entry.id} />
                ))}
              </Menu.ItemGroup>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
    </Group>
  );
};

const NewProjectStartMenuItem = ({ entry }: { entry: LaunchpadStartEntry }) => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const handleSelect = useCallback(() => void navigate({ search: entry.search, to: '/app' }), [entry.search, navigate]);

  return (
    <MenuActionItem
      hint={t(entry.descriptionKey)}
      icon={entry.icon}
      label={t(entry.labelKey)}
      value={entry.id}
      onSelect={handleSelect}
    />
  );
};
