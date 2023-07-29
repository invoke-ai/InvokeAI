import { Box, Flex } from '@chakra-ui/react';
import { CSSObject } from '@emotion/react';
import { motion } from 'framer-motion';
import { FaLink } from 'react-icons/fa';

const sharedConcatLinkStyle: CSSObject = {
  position: 'absolute',
  bg: 'none',
  w: 'full',
  minH: 2,
  borderRadius: 0,
  borderLeft: 'none',
  borderRight: 'none',
  zIndex: 2,
  maskImage:
    'radial-gradient(circle at center, black, black 65%, black 30%, black 15%, transparent)',
};

export default function SDXLConcatLink() {
  return (
    <Flex>
      <Box
        as={motion.div}
        initial={{
          scaleX: 0,
          borderWidth: 0,
          display: 'none',
        }}
        animate={{
          display: ['block', 'block', 'block', 'none'],
          scaleX: [0, 0.25, 0.5, 1],
          borderWidth: [0, 3, 3, 0],
          transition: { duration: 0.37, times: [0, 0.25, 0.5, 1] },
        }}
        sx={{
          top: '1px',
          borderTop: 'none',
          borderColor: 'base.400',
          ...sharedConcatLinkStyle,
          _dark: {
            borderColor: 'accent.500',
          },
        }}
      />
      <Box
        as={motion.div}
        initial={{
          opacity: 0,
          scale: 0,
        }}
        animate={{
          opacity: [0, 1, 1, 1],
          scale: [0, 0.75, 1.5, 1],
          transition: { duration: 0.42, times: [0, 0.25, 0.5, 1] },
        }}
        exit={{
          opacity: 0,
          scale: 0,
        }}
        sx={{
          zIndex: 3,
          position: 'absolute',
          left: '48%',
          top: '3px',
          p: 1,
          borderRadius: 4,
          bg: 'accent.400',
          color: 'base.50',
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
          borderWidth: 0,
          display: 'none',
        }}
        animate={{
          display: ['block', 'block', 'block', 'none'],
          scaleX: [0, 0.25, 0.5, 1],
          borderWidth: [0, 3, 3, 0],
          transition: { duration: 0.37, times: [0, 0.25, 0.5, 1] },
        }}
        sx={{
          top: '17px',
          borderBottom: 'none',
          borderColor: 'base.400',
          ...sharedConcatLinkStyle,
          _dark: {
            borderColor: 'accent.500',
          },
        }}
      />
    </Flex>
  );
}
