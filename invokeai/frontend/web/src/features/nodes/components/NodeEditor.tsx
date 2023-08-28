import { Flex } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { AnimatePresence, motion } from 'framer-motion';
import { memo } from 'react';
import { MdDeviceHub } from 'react-icons/md';
import 'reactflow/dist/style.css';
import AddNodePopover from './flow/AddNodePopover/AddNodePopover';
import { Flow } from './flow/Flow';
import TopLeftPanel from './flow/panels/TopLeftPanel/TopLeftPanel';
import TopCenterPanel from './flow/panels/TopCenterPanel/TopCenterPanel';
import TopRightPanel from './flow/panels/TopRightPanel/TopRightPanel';
import BottomLeftPanel from './flow/panels/BottomLeftPanel/BottomLeftPanel';
import MinimapPanel from './flow/panels/MinimapPanel/MinimapPanel';

const NodeEditor = () => {
  const isReady = useAppSelector((state) => state.nodes.isReady);
  return (
    <Flex
      layerStyle="first"
      sx={{
        position: 'relative',
        width: 'full',
        height: 'full',
        borderRadius: 'base',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <AnimatePresence>
        {isReady && (
          <motion.div
            initial={{
              opacity: 0,
            }}
            animate={{
              opacity: 1,
              transition: { duration: 0.2 },
            }}
            exit={{
              opacity: 0,
              transition: { duration: 0.2 },
            }}
            style={{ position: 'relative', width: '100%', height: '100%' }}
          >
            <Flow />
            <AddNodePopover />
            <TopLeftPanel />
            <TopCenterPanel />
            <TopRightPanel />
            <BottomLeftPanel />
            <MinimapPanel />
          </motion.div>
        )}
      </AnimatePresence>
      <AnimatePresence>
        {!isReady && (
          <motion.div
            initial={{
              opacity: 0,
            }}
            animate={{
              opacity: 1,
              transition: { duration: 0.2 },
            }}
            exit={{
              opacity: 0,
              transition: { duration: 0.2 },
            }}
            style={{ position: 'absolute', width: '100%', height: '100%' }}
          >
            <Flex
              layerStyle="first"
              sx={{
                position: 'relative',
                width: 'full',
                height: 'full',
                borderRadius: 'base',
                alignItems: 'center',
                justifyContent: 'center',
                pointerEvents: 'none',
              }}
            >
              <IAINoContentFallback
                label="Loading Nodes..."
                icon={MdDeviceHub}
              />
            </Flex>
          </motion.div>
        )}
      </AnimatePresence>
    </Flex>
  );
};

export default memo(NodeEditor);
