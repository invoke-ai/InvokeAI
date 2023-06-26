import { useColorMode } from '@chakra-ui/react';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaMoon, FaSun } from 'react-icons/fa';

const ColorModeButton = () => {
  const { colorMode, toggleColorMode } = useColorMode();

  return (
    <IAIIconButton
      aria-label="Toggle Color Mode"
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
