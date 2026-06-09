import { Button, Icon, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { PiCaretDownBold, PiCheckBold } from 'react-icons/pi';

import { useWorkbench } from '../WorkbenchContext';
import { getLayoutPreset, layoutPresets } from '../layoutPresets';
import type { LayoutPresetId } from '../types';

/**
 * Global layout preset registry surfaced as a menu.
 *
 * Replaces the prototype's hand-rolled absolutely-positioned popover (and its
 * open/close `useState`) with a Chakra `Menu`, which handles focus trapping,
 * outside-click dismissal, and keyboard navigation. Presets are global; applying
 * one mutates only the active project's layout state, per the spec.
 */
export const LayoutPresetMenu = () => {
  const { activeProject, dispatch } = useWorkbench();
  const activePreset = getLayoutPreset(activeProject.layout.presetId);

  return (
    <Menu.Root positioning={{ placement: 'bottom-end' }}>
      <Menu.Trigger asChild>
        <Button
          bg="bg.surface"
          borderWidth="1px"
          borderColor="border.emphasis"
          color="fg.default"
          fontSize="xs"
          fontWeight="500"
          justifyContent="space-between"
          size="xs"
          variant="outline"
          w="9rem"
          _hover={{ bg: 'bg.surfaceRaised' }}
        >
          {activePreset.label}
          <Icon as={PiCaretDownBold} boxSize="3" />
        </Button>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <Menu.Content
            bg="bg.surfaceRaised"
            borderWidth="1px"
            borderColor="border.emphasis"
            color="fg.default"
            maxW="18rem"
            rounded="lg"
            shadow="lg"
          >
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
                  {preset.id === activePreset.id ? <Icon as={PiCheckBold} boxSize="3" color="accent.active" /> : null}
                </Menu.Item>
              ))}
            </Menu.ItemGroup>
          </Menu.Content>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
