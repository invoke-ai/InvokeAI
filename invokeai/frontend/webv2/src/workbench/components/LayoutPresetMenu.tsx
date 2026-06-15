import type { LayoutPresetId } from '@workbench/types';

import { Icon, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { getLayoutPreset, layoutPresets } from '@workbench/layoutPresets';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { CheckIcon, ChevronDownIcon } from 'lucide-react';

import { Button } from './ui/Button';

/**
 * Global layout preset registry surfaced as a menu.
 *
 * Replaces the prototype's hand-rolled absolutely-positioned popover (and its
 * open/close `useState`) with a Chakra `Menu`, which handles focus trapping,
 * outside-click dismissal, and keyboard navigation. Presets are global; applying
 * one mutates only the active project's layout state, per the spec.
 */
export const LayoutPresetMenu = () => {
  const activePresetId = useActiveProjectSelector((project) => project.layout.presetId);
  const dispatch = useWorkbenchDispatch();
  const activePreset = getLayoutPreset(activePresetId);

  return (
    <Menu.Root positioning={{ placement: 'bottom-end' }}>
      <Menu.Trigger asChild>
        <Button
          bg="bg.subtle"
          borderWidth="1px"
          borderColor="border.emphasized"
          color="fg"
          fontSize="xs"
          fontWeight="500"
          justifyContent="space-between"
          size="xs"
          variant="outline"
          w="9rem"
          _hover={{ bg: 'bg.muted' }}
        >
          {activePreset.label}
          <Icon as={ChevronDownIcon} boxSize="3" />
        </Button>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <Menu.Content maxW="18rem">
            <Menu.ItemGroup>
              <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                Layout presets
              </Menu.ItemGroupLabel>
              {layoutPresets.map((preset) => (
                <Menu.Item
                  key={preset.id}
                  value={preset.id}
                  onClick={() => dispatch({ presetId: preset.id as LayoutPresetId, type: 'applyPreset' })}
                >
                  <Stack gap="0.5" flex="1" minW="0">
                    <Text fontSize="xs" fontWeight="600">
                      {preset.label}
                    </Text>
                    <Text color="fg.subtle" fontSize="2xs">
                      {preset.description}
                    </Text>
                  </Stack>
                  {preset.id === activePreset.id ? <Icon as={CheckIcon} boxSize="3" color="accent.solid" /> : null}
                </Menu.Item>
              ))}
            </Menu.ItemGroup>
          </Menu.Content>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
