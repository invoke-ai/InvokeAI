import { useColorMode } from '@chakra-ui/react';
import IAIIconButton from 'common/components/IAIIconButton';
import { useTranslation } from 'react-i18next';
import { FaMoon, FaSun } from 'react-icons/fa';

const ColorModeButton = () => {
  const { colorMode, toggleColorMode } = useColorMode();
  const { t } = useTranslation();

  return (
    <IAIIconButton
      aria-label={
        colorMode === 'dark' ? t('common.lightMode') : t('common.darkMode')
      }
      tooltip={
        colorMode === 'dark' ? t('common.lightMode') : t('common.darkMode')
      }
      size="sm"
      icon={
        colorMode === 'dark' ? (
          <FaSun fontSize={19} />
        ) : (
          <FaMoon fontSize={18} />
        )
      }
      onClick={toggleColorMode}
      variant="link"
    />
  );
};

export default ColorModeButton;
