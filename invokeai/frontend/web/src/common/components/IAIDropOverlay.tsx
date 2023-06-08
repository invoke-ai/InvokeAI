import { Flex, Text } from '@chakra-ui/react';
import { motion } from 'framer-motion';
import { memo, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';

type Props = {
  isOver: boolean;
  label?: string;
};

export const IAIDropOverlay = (props: Props) => {
  const { isOver, label = 'Drop' } = props;
  const motionId = useRef(uuidv4());
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
            bg: 'base.900',
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
            top: 0,
            insetInlineStart: 0,
            w: 'full',
            h: 'full',
            opacity: 1,
            borderWidth: 2,
            borderColor: isOver ? 'base.200' : 'base.500',
            borderRadius: 'base',
            borderStyle: 'dashed',
            transitionProperty: 'common',
            transitionDuration: '0.1s',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Text
            sx={{
              fontSize: '2xl',
              fontWeight: 600,
              transform: isOver ? 'scale(1.1)' : 'scale(1)',
              color: isOver ? 'base.100' : 'base.500',
              transitionProperty: 'common',
              transitionDuration: '0.1s',
            }}
          >
            {label}
          </Text>
        </Flex>
      </Flex>
    </motion.div>
  );
};

export default memo(IAIDropOverlay);
