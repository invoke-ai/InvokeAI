import { useDisclosure } from '@chakra-ui/react';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { InvMenuItem } from 'common/components/InvMenu/InvMenuItem';
import { InvMenuList } from 'common/components/InvMenu/InvMenuList';
import {
  InvMenu,
  InvMenuButton,
  InvMenuGroup,
} from 'common/components/InvMenu/wrapper';
import { useGlobalMenuClose } from 'common/hooks/useGlobalMenuClose';
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
  PiToggleRightFill,
} from 'react-icons/pi';
import { RiDiscordFill, RiGithubFill, RiSettings4Line } from 'react-icons/ri';

import SettingsModal from './SettingsModal';
const SettingsMenu = () => {
  const { t } = useTranslation();
  const { isOpen, onOpen, onClose } = useDisclosure();
  useGlobalMenuClose(onClose);

  const isBugLinkEnabled = useFeatureStatus('bugLink').isFeatureEnabled;
  const isDiscordLinkEnabled = useFeatureStatus('discordLink').isFeatureEnabled;
  const isGithubLinkEnabled = useFeatureStatus('githubLink').isFeatureEnabled;

  return (
    <InvMenu isOpen={isOpen} onOpen={onOpen} onClose={onClose}>
      <InvMenuButton
        as={InvIconButton}
        variant="link"
        aria-label={t('accessibility.menu')}
        icon={<RiSettings4Line fontSize={20} />}
        boxSize={8}
      />

      <InvMenuList>
        <InvMenuGroup title={t('common.communityLabel')}>
          {isGithubLinkEnabled && (
            <InvMenuItem
              as="a"
              href={githubLink}
              target="_blank"
              icon={<RiGithubFill />}
            >
              {t('common.githubLabel')}
            </InvMenuItem>
          )}
          {isBugLinkEnabled && (
            <InvMenuItem
              as="a"
              href={`${githubLink}/issues`}
              target="_blank"
              icon={<PiBugBeetleBold />}
            >
              {t('common.reportBugLabel')}
            </InvMenuItem>
          )}
          {isDiscordLinkEnabled && (
            <InvMenuItem
              as="a"
              href={discordLink}
              target="_blank"
              icon={<RiDiscordFill />}
            >
              {t('common.discordLabel')}
            </InvMenuItem>
          )}
        </InvMenuGroup>
        <InvMenuGroup title={t('common.settingsLabel')}>
          <HotkeysModal>
            <InvMenuItem as="button" icon={<PiKeyboardBold />}>
              {t('common.hotkeysLabel')}
            </InvMenuItem>
          </HotkeysModal>
          <SettingsModal>
            <InvMenuItem as="button" icon={<PiToggleRightFill />}>
              {t('common.settingsLabel')}
            </InvMenuItem>
          </SettingsModal>
        </InvMenuGroup>
        <InvMenuGroup title={t('accessibility.about')}>
          <AboutModal>
            <InvMenuItem as="button" icon={<PiInfoBold />}>
              {t('accessibility.about')}
            </InvMenuItem>
          </AboutModal>
        </InvMenuGroup>
      </InvMenuList>
    </InvMenu>
  );
};

export default memo(SettingsMenu);
