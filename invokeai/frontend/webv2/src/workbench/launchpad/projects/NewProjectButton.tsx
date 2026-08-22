import type { BuiltInLayoutPresetId } from '@workbench/layoutContracts';
import type { LucideIcon } from 'lucide-react';

import { Icon, Menu, Portal } from '@chakra-ui/react';
// Concrete modules, not the barrel: `@platform/ui` sits at its direct-importer
// budget, and this component needs four of its exports.
import { Button, IconButton } from '@platform/ui/Button';
import { Group } from '@platform/ui/Group';
import { MenuContent } from '@platform/ui/Menu';
import { Link } from '@tanstack/react-router';
import { BUILT_IN_LAYOUT_PRESET_LABELS, LAUNCHPAD_LAYOUT_IDS } from '@workbench/launchpad/intents';
import { ChevronDownIcon, LayersIcon, PlusIcon, TypeIcon, WorkflowIcon } from 'lucide-react';
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
 * Labels come from the shared id/label map that `layoutPresets` also builds
 * from, so the menu and the editor's preset strip cannot disagree about what a
 * layout is called — without the Launchpad having to load the preset table's
 * three full widget-region snapshots. Icons are mapped locally for the same
 * reason: `resolveLayoutPresetIcon` exists for the custom-preset picker and
 * carries its whole curated catalogue.
 */

const NEW_PROJECT_SEARCH = { new: true } as const;
const MENU_POSITIONING = { placement: 'bottom-end' } as const;

const LAYOUT_ICONS: Record<BuiltInLayoutPresetId, LucideIcon> = {
  automate: WorkflowIcon,
  compose: TypeIcon,
  edit: LayersIcon,
};

interface NewProjectLayoutItem {
  icon: LucideIcon;
  id: BuiltInLayoutPresetId;
  label: string;
  search: { new: true; preset: BuiltInLayoutPresetId };
}

const LAYOUT_ITEMS: NewProjectLayoutItem[] = LAUNCHPAD_LAYOUT_IDS.map((id) => ({
  icon: LAYOUT_ICONS[id],
  id,
  label: BUILT_IN_LAYOUT_PRESET_LABELS[id],
  search: { new: true, preset: id },
}));

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
