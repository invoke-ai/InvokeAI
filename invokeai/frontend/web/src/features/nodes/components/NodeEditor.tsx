import 'reactflow/dist/style.css';

import { Flex } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import TopPanel from 'features/nodes/components/flow/panels/TopPanel/TopPanel';
import { AnimatePresence, motion } from 'framer-motion';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { MdDeviceHub } from 'react-icons/md';

import AddNodePopover from './flow/AddNodePopover/AddNodePopover';
import { Flow } from './flow/Flow';
import BottomLeftPanel from './flow/panels/BottomLeftPanel/BottomLeftPanel';
import MinimapPanel from './flow/panels/MinimapPanel/MinimapPanel';

const NodeEditor = () => {
  const isReady = useAppSelector((state) => state.nodes.isReady);
  const { t } = useTranslation();
  return (
    <Flex
      layerStyle="first"
      position="relative"
      width="full"
      height="full"
      borderRadius="base"
      alignItems="center"
      justifyContent="center"
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
            <TopPanel />
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
              position="relative"
              width="full"
              height="full"
              borderRadius="base"
              alignItems="center"
              justifyContent="center"
              pointerEvents="none"
            >
              <IAINoContentFallback
                label={t('nodes.loadingNodes')}
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
