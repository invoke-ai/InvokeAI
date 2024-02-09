import {
  Button,
  Flex,
  Menu,
  MenuButton,
  MenuItem,
  MenuList,
  Text,
} from '@invoke-ai/ui-library';
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
      { value: 'view', label: t('nodes.viewMode'), icon: <MdBolt /> },
      { value: 'edit', label: t('nodes.editMode'), icon: <PiPencil /> },
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
          rightIcon={<PiCaretDownBold fontSize="xs" />}
          textAlign="left"
        >
          <Text fontSize="sm">{value?.label}</Text>
        </MenuButton>
        <MenuList>
          <MenuItem fontSize="sm" icon={<MdBolt />} onClick={handleClickMode.bind(null, 'view')}>
            View Mode
          </MenuItem>
          <MenuItem fontSize="sm" icon={<PiPencil />} onClick={handleClickMode.bind(null, 'edit')}>
            Edit Mode
          </MenuItem>
        </MenuList>
      </Menu>
    </Flex>
  );
};
