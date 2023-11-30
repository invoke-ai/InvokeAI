import {
  Flex,
  Menu,
  MenuButton,
  MenuGroup,
  MenuItem,
  MenuList,
} from '@chakra-ui/react';
import IAIIconButton from 'common/components/IAIIconButton';
import WorkflowLibraryButton from 'features/nodes/components/flow/WorkflowLibrary/WorkflowLibraryButton';
import { memo } from 'react';
import { FaEllipsis } from 'react-icons/fa6';
import WorkflowEditorSettings from './WorkflowEditorSettings';
import { MENU_LIST_MOTION_PROPS as MENU_LIST_MOTION_PROPS } from 'theme/components/menu';
import { useTranslation } from 'react-i18next';
import { useDownloadWorkflow } from 'features/nodes/hooks/useDownloadWorkflow';
import { FaDownload, FaSave } from 'react-icons/fa';
import { useSaveWorkflow } from 'features/nodes/hooks/useSaveWorkflow';

const TopRightPanel = () => {
  const { t } = useTranslation();
  const downloadWorkflow = useDownloadWorkflow();
  const saveWorkflow = useSaveWorkflow();
  return (
    <Flex sx={{ gap: 2, position: 'absolute', top: 2, insetInlineEnd: 2 }}>
      <WorkflowEditorSettings />
      <WorkflowLibraryButton />
      <Menu>
        <MenuButton as={IAIIconButton} icon={<FaEllipsis />} />
        <MenuList motionProps={MENU_LIST_MOTION_PROPS}>
          <MenuItem onClick={saveWorkflow} icon={<FaSave />}>
            {t('workflows.saveWorkflow')}
          </MenuItem>
          <MenuItem onClick={downloadWorkflow} icon={<FaDownload />}>
            {t('workflows.downloadWorkflow')}
          </MenuItem>
          {/* <MenuGroup title={t('common.settingsLabel')}>
            <HotkeysModal>
              <MenuItem as="button" icon={<FaKeyboard />}>
                {t('common.hotkeysLabel')}
              </MenuItem>
            </HotkeysModal>
            <SettingsModal>
              <MenuItem as="button" icon={<FaCog />}>
                {t('common.settingsLabel')}
              </MenuItem>
            </SettingsModal>
          </MenuGroup> */}
        </MenuList>
      </Menu>
    </Flex>
  );
};

export default memo(TopRightPanel);
