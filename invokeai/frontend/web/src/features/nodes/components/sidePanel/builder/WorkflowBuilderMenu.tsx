import { IconButton, Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/nanostores/store';
import { useResetAllNodeFields } from 'features/nodes/components/sidePanel/builder/use-reset-all-node-fields';
import { formReset } from 'features/nodes/store/workflowSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiDotsThreeBold, PiTrashBold } from 'react-icons/pi';

export const WorkflowBuilderEditMenu = memo(() => {
  const { t } = useTranslation();
  const store = useAppStore();
  const resetAllNodeFields = useResetAllNodeFields();
  const deleteAllElements = useCallback(() => {
    store.dispatch(formReset());
  }, [store]);
  return (
    <Menu placement="bottom-end">
      <MenuButton as={IconButton} icon={<PiDotsThreeBold />} variant="ghost" size="sm" />
      <MenuList>
        <MenuItem icon={<PiArrowCounterClockwiseBold />} onClick={resetAllNodeFields}>
          {t('workflows.builder.resetAllNodeFields')}
        </MenuItem>
        <MenuItem isDestructive icon={<PiTrashBold />} onClick={deleteAllElements}>
          {t('workflows.builder.deleteAllElements')}
        </MenuItem>
      </MenuList>
    </Menu>
  );
});
WorkflowBuilderEditMenu.displayName = 'WorkflowBuilderEditMenu';
