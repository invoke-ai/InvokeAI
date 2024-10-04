import { Flex, Icon, MenuItem, Text, Tooltip } from '@invoke-ai/ui-library';
import { useTranslation } from 'react-i18next';
import type { IconType } from 'react-icons';
import { PiArrowUpBold } from 'react-icons/pi';

export const SettingsUpsellMenuItem = ({ menuText, menuIcon }: { menuText: string; menuIcon: IconType }) => {
  const { t } = useTranslation();

  return (
    <Tooltip label={t('upsell.professionalUpsell')} placement="right" gutter={16}>
      <MenuItem as="a" href="http://invoke.com/pricing" target="_blank" icon={menuIcon({})}>
        <Flex gap="1" alignItems="center" justifyContent="space-between">
          <Text>{menuText}</Text>
          <Icon as={PiArrowUpBold} color="invokeYellow.500" />
        </Flex>
      </MenuItem>
    </Tooltip>
  );
};
