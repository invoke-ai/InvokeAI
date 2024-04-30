import { Box, Flex } from '@invoke-ai/ui-library';
import CurrentImageDisplay from 'features/gallery/components/CurrentImage/CurrentImageDisplay';
import InitialImageDisplay from 'features/parameters/components/ImageToImage/InitialImageDisplay';
import ResizeHandle from 'features/ui/components/tabs/ResizeHandle';
import { usePanelStorage } from 'features/ui/hooks/usePanelStorage';
import type { CSSProperties } from 'react';
import { memo, useCallback, useRef } from 'react';
import type { ImperativePanelGroupHandle } from 'react-resizable-panels';
import { Panel, PanelGroup } from 'react-resizable-panels';

const panelGroupStyles: CSSProperties = {
  height: '100%',
  width: '100%',
};
const panelStyles: CSSProperties = {
  position: 'relative',
};

const ImageToImageTab = () => {
  const panelGroupRef = useRef<ImperativePanelGroupHandle>(null);

  const handleDoubleClickHandle = useCallback(() => {
    if (!panelGroupRef.current) {
      return;
    }
    panelGroupRef.current.setLayout([50, 50]);
  }, []);

  const panelStorage = usePanelStorage();

  return (
    <Box w="full" h="full">
      <PanelGroup
        ref={panelGroupRef}
        autoSaveId="imageTab.content"
        direction="horizontal"
        style={panelGroupStyles}
        storage={panelStorage}
      >
        <Panel id="imageTab.content.initImage" order={0} defaultSize={50} minSize={25} style={panelStyles}>
          <InitialImageDisplay />
        </Panel>
        <ResizeHandle orientation="vertical" onDoubleClick={handleDoubleClickHandle} />
        <Panel id="imageTab.content.selectedImage" order={1} defaultSize={50} minSize={25}>
          <Box layerStyle="first" position="relative" w="full" h="full" p={2} borderRadius="base">
            <Flex w="full" h="full">
              <CurrentImageDisplay />
            </Flex>
          </Box>
        </Panel>
      </PanelGroup>
    </Box>
  );
};

export default memo(ImageToImageTab);
