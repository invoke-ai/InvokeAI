import { Box } from '@chakra-ui/react';
import InitialImageDisplay from 'features/parameters/components/Parameters/ImageToImage/InitialImageDisplay';
import { usePanelStorage } from 'features/ui/hooks/usePanelStorage';
import { memo, useCallback, useRef } from 'react';
import {
  ImperativePanelGroupHandle,
  Panel,
  PanelGroup,
} from 'react-resizable-panels';
import ResizeHandle from '../ResizeHandle';
import TextToImageTabMain from '../TextToImage/TextToImageTabMain';

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
    <Box sx={{ w: 'full', h: 'full' }}>
      <PanelGroup
        ref={panelGroupRef}
        autoSaveId="imageTab.content"
        direction="horizontal"
        style={{ height: '100%', width: '100%' }}
        storage={panelStorage}
        units="percentages"
      >
        <Panel
          id="imageTab.content.initImage"
          order={0}
          defaultSize={50}
          minSize={25}
          style={{ position: 'relative' }}
        >
          <InitialImageDisplay />
        </Panel>
        <ResizeHandle onDoubleClick={handleDoubleClickHandle} />
        <Panel
          id="imageTab.content.selectedImage"
          order={1}
          defaultSize={50}
          minSize={25}
        >
          <TextToImageTabMain />
        </Panel>
      </PanelGroup>
    </Box>
  );
};

export default memo(ImageToImageTab);
