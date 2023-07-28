import { Box, Flex } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { motion } from 'framer-motion';
import { FaLink } from 'react-icons/fa';

export default function SDXLConcatLink() {
  const shouldConcatSDXLStylePrompt = useAppSelector(
    (state: RootState) => state.sdxl.shouldConcatSDXLStylePrompt
  );
  return (
    shouldConcatSDXLStylePrompt && (
      <Flex
        sx={{
          h: 0.5,
          placeContent: 'center',
          gap: 2,
          px: 2,
        }}
      >
        <Box
          as={motion.div}
          initial={{
            scaleX: 0,
          }}
          animate={{ scaleX: 1, transition: { duration: 0.3 } }}
          sx={{
            bg: 'accent.300',
            h: 0.5,
            w: 'full',
            borderRadius: 9999,
            transformOrigin: 'right',
            _dark: {
              bg: 'accent.500',
            },
          }}
        />
        <Box
          as={motion.div}
          initial={{
            opacity: 0,
          }}
          animate={{
            opacity: 1,
            transition: { duration: 0.3 },
          }}
          zIndex={2}
          mt={-1.5}
        >
          <FaLink color="var(--invokeai-colors-accent-400)" />
        </Box>
        <Box
          as={motion.div}
          initial={{
            scaleX: 0,
          }}
          animate={{ scaleX: 1, transition: { duration: 0.3 } }}
          sx={{
            bg: 'accent.300',
            h: 0.5,
            w: 'full',
            borderRadius: 9999,
            transformOrigin: 'left',
            _dark: {
              bg: 'accent.500',
            },
          }}
        />
      </Flex>
    )
  );
}
