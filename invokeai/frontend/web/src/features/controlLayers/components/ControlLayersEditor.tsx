/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { ControlLayersToolbar } from 'features/controlLayers/components/ControlLayersToolbar';
import { StageComponent } from 'features/controlLayers/components/StageComponent';
import { memo } from 'react';

export const ControlLayersEditor = memo(() => {
  return (
    <Flex
      position="relative"
      flexDirection="column"
      height="100%"
      width="100%"
      rowGap={4}
      alignItems="center"
      justifyContent="center"
    >
      <ControlLayersToolbar />
      <StageComponent />
    </Flex>
  );
});

ControlLayersEditor.displayName = 'ControlLayersEditor';
