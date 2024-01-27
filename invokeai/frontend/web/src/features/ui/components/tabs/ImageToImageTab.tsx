import { Box } from '@invoke-ai/ui-library';
import InitialImageDisplay from 'features/parameters/components/ImageToImage/InitialImageDisplay';
import ResizeHandle from 'features/ui/components/tabs/ResizeHandle';
import TextToImageTabMain from 'features/ui/components/tabs/TextToImageTab';
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
          <TextToImageTabMain />
        </Panel>
      </PanelGroup>
    </Box>
  );
};

export default memo(ImageToImageTab);
