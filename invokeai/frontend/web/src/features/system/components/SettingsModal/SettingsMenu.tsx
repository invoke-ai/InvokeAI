import {
  IconButton,
  Menu,
  MenuButton,
  MenuGroup,
  MenuItem,
  MenuList,
  useDisclosure,
  useGlobalMenuClose,
} from '@invoke-ai/ui-library';
import AboutModal from 'features/system/components/AboutModal/AboutModal';
import HotkeysModal from 'features/system/components/HotkeysModal/HotkeysModal';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { discordLink, githubLink } from 'features/system/store/constants';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiBugBeetleBold,
  PiInfoBold,
  PiKeyboardBold,
  PiShareNetworkFill,
  PiToggleRightFill,
  PiUsersBold,
} from 'react-icons/pi';
import { RiDiscordFill, RiGithubFill, RiSettings4Line } from 'react-icons/ri';

import SettingsModal from './SettingsModal';
import { SettingsUpsellMenuItem } from './SettingsUpsellMenuItem';
const SettingsMenu = () => {
  const { t } = useTranslation();
  const { isOpen, onOpen, onClose } = useDisclosure();
  useGlobalMenuClose(onClose);

  const isBugLinkEnabled = useFeatureStatus('bugLink');
  const isDiscordLinkEnabled = useFeatureStatus('discordLink');
  const isGithubLinkEnabled = useFeatureStatus('githubLink');

  return (
    <Menu isOpen={isOpen} onOpen={onOpen} onClose={onClose} autoSelect={false}>
      <MenuButton
        as={IconButton}
        variant="link"
        aria-label={t('accessibility.menu')}
        icon={<RiSettings4Line fontSize={20} />}
        boxSize={8}
      />
      <MenuList zIndex={1}>
        <MenuGroup title={t('upsell.professional')}>
          <SettingsUpsellMenuItem menuText={t('upsell.inviteTeammates')} menuIcon={PiUsersBold} />
          <SettingsUpsellMenuItem menuText={t('upsell.shareAccess')} menuIcon={PiShareNetworkFill} />
        </MenuGroup>

        <MenuGroup title={t('common.communityLabel')}>
          {isGithubLinkEnabled && (
            <MenuItem as="a" href={githubLink} target="_blank" icon={<RiGithubFill />}>
              {t('common.githubLabel')}
            </MenuItem>
          )}
          {isBugLinkEnabled && (
            <MenuItem as="a" href={`${githubLink}/issues`} target="_blank" icon={<PiBugBeetleBold />}>
              {t('common.reportBugLabel')}
            </MenuItem>
          )}
          {isDiscordLinkEnabled && (
            <MenuItem as="a" href={discordLink} target="_blank" icon={<RiDiscordFill />}>
              {t('common.discordLabel')}
            </MenuItem>
          )}
        </MenuGroup>
        <MenuGroup title={t('common.settingsLabel')}>
          <HotkeysModal>
            <MenuItem as="button" icon={<PiKeyboardBold />}>
              {t('common.hotkeysLabel')}
            </MenuItem>
          </HotkeysModal>
          <SettingsModal>
            <MenuItem as="button" icon={<PiToggleRightFill />}>
              {t('common.settingsLabel')}
            </MenuItem>
          </SettingsModal>
        </MenuGroup>
        <MenuGroup title={t('accessibility.about')}>
          <AboutModal>
            <MenuItem as="button" icon={<PiInfoBold />}>
              {t('accessibility.about')}
            </MenuItem>
          </AboutModal>
        </MenuGroup>
      </MenuList>
    </Menu>
  );
};

export default memo(SettingsMenu);
