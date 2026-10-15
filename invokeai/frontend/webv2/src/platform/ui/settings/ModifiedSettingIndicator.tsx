import { Box } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';

export const ModifiedSettingIndicator = ({ label }: { label?: string }) => {
  const { t } = useTranslation();
  const description = label
    ? t('settingsDialog.changedSetting', { setting: label })
    : t('settingsDialog.changedFromDefault');
  return (
    <Box
      as="span"
      role="img"
      aria-label={description}
      title={description}
      boxSize="1.5"
      flexShrink={0}
      borderRadius="full"
      bg="accent.fg"
    />
  );
};
