import { Box, Flex } from '@chakra-ui/react';
import { motion } from 'framer-motion';
import { FaLink } from 'react-icons/fa';

export default function SDXLConcatLink() {
  return (
    <Flex
      sx={{
        h: 0.5,
        placeContent: 'center',
        gap: 2,
        flexDirection: 'column',
      }}
    >
      <Box
        as={motion.div}
        initial={{
          scaleX: 0,
          borderRadius: 0,
          borderWidth: 0,
          display: 'none',
        }}
        animate={{
          display: ['block', 'block', 'block', 'none'],
          scaleX: [0, 0.25, 0.5, 1],
          borderRadius: [0, 0, 0, 3],
          borderWidth: [0, 3, 3, 0],
          transition: { duration: 0.5, times: [0, 0.25, 0.5, 1] },
        }}
        sx={{
          position: 'absolute',
          top: '1px',
          bg: 'none',
          w: 'full',
          minH: 2,
          borderTop: 'none',
          borderTopRadius: 0,
          borderColor: 'accent.300',
          zIndex: 2,
          _dark: {
            borderColor: 'accent.600',
          },
        }}
      />
      <Box
        as={motion.div}
        initial={{
          opacity: 0,
          scale: 0,
          rotate: 0,
        }}
        animate={{
          rotate: 360,
          opacity: [0, 1, 1, 1],
          scale: [0, 0.75, 1.5, 1],
          transition: { duration: 0.6, times: [0, 0.25, 0.5, 1] },
        }}
        sx={{
          zIndex: 3,
          position: 'absolute',
          left: '48%',
          top: '3px',
          p: 1,
          borderRadius: 4,
          bg: 'accent.200',
          _dark: {
            bg: 'accent.500',
          },
        }}
      >
        <FaLink size={12} />
      </Box>
      <Box
        as={motion.div}
        initial={{
          scaleX: 0,
          borderRadius: 0,
          borderWidth: 0,
          display: 'none',
        }}
        animate={{
          display: ['block', 'block', 'block', 'none'],
          scaleX: [0, 0.25, 0.5, 1],
          borderRadius: [0, 0, 0, 3],
          borderWidth: [0, 3, 3, 0],
          transition: { duration: 0.5, times: [0, 0.25, 0.5, 1] },
        }}
        sx={{
          position: 'absolute',
          top: '17px',
          bg: 'none',
          w: 'full',
          minH: 2,
          borderBottom: 'none',
          borderBottomRadius: 0,
          borderColor: 'accent.300',
          zIndex: 2,
          _dark: {
            borderColor: 'accent.600',
          },
        }}
      />
    </Flex>
  );
}
