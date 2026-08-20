import type { BuiltInLayoutPresetId } from '@workbench/layoutContracts';
import type { LucideIcon } from 'lucide-react';

import { Icon, Menu, Portal } from '@chakra-ui/react';
import { Button, Group, IconButton, MenuContent } from '@platform/ui';
import { Link } from '@tanstack/react-router';
import { LAUNCHPAD_LAYOUT_IDS } from '@workbench/launchpad/intents';
import { layoutPresets } from '@workbench/layoutPresets';
import { ChevronDownIcon, LayersIcon, LayoutGridIcon, PlusIcon, TypeIcon, WorkflowIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

/**
 * New project, plus the arrangement to open it in.
 *
 * The plain click keeps the old behaviour — a draft in whatever preset the
 * account last used — because most of the time the arrangement is not the
 * decision being made. The caret is for when it is: it names the three shipped
 * layouts so starting from "Edit" does not mean opening a draft and then
 * rearranging it.
 *
 * Labels come from the preset table itself rather than a parallel list, so the
 * menu and the editor's preset strip can never disagree about what a layout is
 * called. Icons are mapped locally: `resolveLayoutPresetIcon` exists for the
 * custom-preset picker and carries its whole curated catalogue, which is a
 * lot of Lucide to pull onto the Launchpad for three entries.
 */

const NEW_PROJECT_SEARCH = { new: true } as const;
const MENU_POSITIONING = { placement: 'bottom-end' } as const;

const LAYOUT_ICONS: Record<string, LucideIcon> = {
  layers: LayersIcon,
  type: TypeIcon,
  workflow: WorkflowIcon,
};

interface NewProjectLayoutItem {
  icon: LucideIcon;
  id: BuiltInLayoutPresetId;
  label: string;
  search: { new: true; preset: BuiltInLayoutPresetId };
}

const LAYOUT_ITEMS: NewProjectLayoutItem[] = LAUNCHPAD_LAYOUT_IDS.flatMap((id) => {
  const preset = layoutPresets.find((candidate) => candidate.id === id);

  return preset
    ? [
        {
          icon: LAYOUT_ICONS[preset.iconId ?? ''] ?? LayoutGridIcon,
          id,
          label: preset.label,
          search: { new: true, preset: id },
        },
      ]
    : [];
});

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
          <IconButton aria-label={t('projects.newProjectWithLayout')} size="xs" variant={variant}>
            <Icon as={ChevronDownIcon} boxSize="3.5" />
          </IconButton>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="12rem">
              <Menu.ItemGroup>
                <Menu.ItemGroupLabel>{t('projects.newProjectWithLayout')}</Menu.ItemGroupLabel>
                {LAYOUT_ITEMS.map((item) => (
                  <NewProjectLayoutMenuItem key={item.id} item={item} />
                ))}
              </Menu.ItemGroup>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
    </Group>
  );
};

const NewProjectLayoutMenuItem = ({ item }: { item: NewProjectLayoutItem }) => (
  <Menu.Item asChild value={item.id}>
    <Link search={item.search} to="/app">
      <Icon as={item.icon} boxSize="3.5" color="fg.subtle" />
      <Menu.ItemText fontSize="xs">{item.label}</Menu.ItemText>
    </Link>
  </Menu.Item>
);
