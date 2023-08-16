import { Flex } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ResizeHandle from 'features/ui/components/tabs/ResizeHandle';
import { memo, useState } from 'react';
import { MdDeviceHub } from 'react-icons/md';
import { Panel, PanelGroup } from 'react-resizable-panels';
import 'reactflow/dist/style.css';
import NodeEditorPanelGroup from './panel/NodeEditorPanelGroup';
import { Flow } from './Flow';
import { AnimatePresence, motion } from 'framer-motion';

const NodeEditor = () => {
  const [isPanelCollapsed, setIsPanelCollapsed] = useState(false);
  const isReady = useAppSelector((state) => state.nodes.isReady);
  return (
    <PanelGroup
      id="node-editor"
      autoSaveId="node-editor"
      direction="horizontal"
      style={{ height: '100%', width: '100%' }}
    >
      <Panel
        id="node-editor-panel-group"
        collapsible
        onCollapse={setIsPanelCollapsed}
        minSize={25}
      >
        <NodeEditorPanelGroup />
      </Panel>
      <ResizeHandle
        collapsedDirection={isPanelCollapsed ? 'left' : undefined}
      />
      <Panel id="node-editor-content">
        <Flex
          layerStyle={'first'}
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
                style={{ width: '100%', height: '100%' }}
              >
                <Flow />
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
                  layerStyle={'first'}
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
      </Panel>
    </PanelGroup>
  );
};

export default memo(NodeEditor);
