import { useDisclosure } from '@chakra-ui/react';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { InvMenuItem } from 'common/components/InvMenu/InvMenuItem';
import { InvMenuList } from 'common/components/InvMenu/InvMenuList';
import {
  InvMenu,
  InvMenuButton,
  InvMenuGroup,
} from 'common/components/InvMenu/wrapper';
import { useGlobalMenuCloseTrigger } from 'common/hooks/useGlobalMenuCloseTrigger';
import HotkeysModal from 'features/system/components/HotkeysModal/HotkeysModal';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiBugBeetleBold,
  PiKeyboardBold,
  PiToggleRightFill,
} from 'react-icons/pi';
import { RiDiscordFill, RiGithubFill, RiSettings4Line } from 'react-icons/ri';

import SettingsModal from './SettingsModal';
const SettingsMenu = () => {
  const { t } = useTranslation();
  const { isOpen, onOpen, onClose } = useDisclosure();
  useGlobalMenuCloseTrigger(onClose);

  const isBugLinkEnabled = useFeatureStatus('bugLink').isFeatureEnabled;
  const isDiscordLinkEnabled = useFeatureStatus('discordLink').isFeatureEnabled;
  const isGithubLinkEnabled = useFeatureStatus('githubLink').isFeatureEnabled;

  const githubLink = 'http://github.com/invoke-ai/InvokeAI';
  const discordLink = 'https://discord.gg/ZmtBAhwWhy';

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
      </InvMenuList>
    </InvMenu>
  );
};

export default memo(SettingsMenu);
