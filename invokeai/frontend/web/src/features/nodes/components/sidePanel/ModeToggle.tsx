import { Button, Flex, Icon, Menu, MenuButton, MenuItem, MenuList, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { WorkflowMode } from 'features/nodes/store/types';
import { workflowModeChanged } from 'features/nodes/store/workflowSlice';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { MdBolt } from 'react-icons/md';
import { PiCaretDownBold, PiPencil } from 'react-icons/pi';

export const ModeToggle = () => {
  const dispatch = useAppDispatch();
  const mode = useAppSelector((s) => s.workflow.mode);
  const { t } = useTranslation();

  const modeOptions = useMemo(() => {
    return [
      {
        value: 'view',
        label: t('nodes.viewMode'),
        icon: <MdBolt />,
        description: 'Use the configured linear Workflow in the studio',
      },
      {
        value: 'edit',
        label: t('nodes.editMode'),
        icon: <PiPencil />,
        description: 'Update and edit the workflow in the Workflow Editor',
      },
    ];
  }, [t]);

  const handleClickMode = useCallback(
    (selectedMode: WorkflowMode) => {
      dispatch(workflowModeChanged(selectedMode));
    },
    [dispatch]
  );

  const value = useMemo(() => modeOptions.find((o) => o.value === mode), [mode, modeOptions]);

  return (
    <Flex justifyContent="flex-end">
      <Menu>
        <MenuButton
          size="md"
          as={Button}
          colorScheme="invokeBlue"
          rightIcon={<PiCaretDownBold size="10" />}
          textAlign="left"
        >
          <Flex gap="1">
            {value?.icon}
            <Text fontSize="sm">{value?.label}</Text>
          </Flex>
        </MenuButton>
        <MenuList maxW="200px">
          {modeOptions.map((option) => (
            <MenuItem
              key={option.value}
              fontSize="sm"
              icon={option.icon}
              onClick={handleClickMode.bind(null, option.value as WorkflowMode)}
              alignItems="flex-start"
            >
              <Flex flexDir="column">
                <Text mt="-2px">{option.label}</Text>
                <Text fontSize="xs">{option.description}</Text>
              </Flex>
            </MenuItem>
          ))}
        </MenuList>
      </Menu>
    </Flex>
  );
};
