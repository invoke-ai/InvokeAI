import { useColorMode, useToken } from '@chakra-ui/react';
import { motion } from 'framer-motion';
import { mode } from 'theme/util/mode';

export const SelectedItemOverlay = () => {
  const [accent400, accent500] = useToken('colors', [
    'accent.400',
    'accent.500',
  ]);

  const { colorMode } = useColorMode();

  return (
    <motion.div
      initial={{
        opacity: 0,
      }}
      animate={{
        opacity: 1,
        transition: { duration: 0.1 },
      }}
      exit={{
        opacity: 0,
        transition: { duration: 0.1 },
      }}
      style={{
        position: 'absolute',
        top: 0,
        insetInlineStart: 0,
        width: '100%',
        height: '100%',
        boxShadow: `inset 0px 0px 0px 2px ${mode(
          accent400,
          accent500
        )(colorMode)}`,
        borderRadius: 'var(--invokeai-radii-base)',
      }}
    />
  );
};
