import { Box, Flex, useColorMode } from '@chakra-ui/react';
import { motion } from 'framer-motion';
import { ReactNode, memo, useRef } from 'react';
import { mode } from 'theme/util/mode';
import { v4 as uuidv4 } from 'uuid';

type Props = {
  isOver: boolean;
  label?: ReactNode;
};

export const IAIDropOverlay = (props: Props) => {
  const { isOver, label = 'Drop' } = props;
  const motionId = useRef(uuidv4());
  const { colorMode } = useColorMode();
  return (
    <motion.div
      key={motionId.current}
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
    >
      <Flex
        sx={{
          position: 'absolute',
          top: 0,
          insetInlineStart: 0,
          w: 'full',
          h: 'full',
        }}
      >
        <Flex
          sx={{
            position: 'absolute',
            top: 0,
            insetInlineStart: 0,
            w: 'full',
            h: 'full',
            bg: mode('base.700', 'base.900')(colorMode),
            opacity: 0.7,
            borderRadius: 'base',
            alignItems: 'center',
            justifyContent: 'center',
            transitionProperty: 'common',
            transitionDuration: '0.1s',
          }}
        />

        <Flex
          sx={{
            position: 'absolute',
            top: 0.5,
            insetInlineStart: 0.5,
            insetInlineEnd: 0.5,
            bottom: 0.5,
            opacity: 1,
            borderWidth: 2,
            borderColor: isOver
              ? mode('base.50', 'base.50')(colorMode)
              : mode('base.200', 'base.300')(colorMode),
            borderRadius: 'lg',
            borderStyle: 'dashed',
            transitionProperty: 'common',
            transitionDuration: '0.1s',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Box
            sx={{
              fontSize: '2xl',
              fontWeight: 600,
              transform: isOver ? 'scale(1.1)' : 'scale(1)',
              color: isOver
                ? mode('base.50', 'base.50')(colorMode)
                : mode('base.200', 'base.300')(colorMode),
              transitionProperty: 'common',
              transitionDuration: '0.1s',
            }}
          >
            {label}
          </Box>
        </Flex>
      </Flex>
    </motion.div>
  );
};

export default memo(IAIDropOverlay);
