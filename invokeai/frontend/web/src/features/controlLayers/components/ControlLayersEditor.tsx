/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { ControlLayersToolbar } from 'features/controlLayers/components/ControlLayersToolbar';
import { StageComponent } from 'features/controlLayers/components/StageComponent';
import { StagingAreaToolbar } from 'features/controlLayers/components/StagingArea/StagingAreaToolbar';
import { memo } from 'react';

export const ControlLayersEditor = memo(() => {
  return (
    <Flex
      position="relative"
      flexDirection="column"
      height="100%"
      width="100%"
      gap={2}
      alignItems="center"
      justifyContent="center"
    >
      <ControlLayersToolbar />
      <StageComponent />
      <Flex position="absolute" bottom={2} gap={2} align="center" justify="center">
        <StagingAreaToolbar />
      </Flex>
      {/* <Flex position="absolute" top={0} right={0} bottom={0} left={0} align="center" justify="center">
        <CanvasResizer />
      </Flex> */}
    </Flex>
  );
});

ControlLayersEditor.displayName = 'ControlLayersEditor';
