import { ButtonGroup } from '@invoke-ai/ui-library';
import { ToolBboxButton } from 'features/controlLayers/components/Tool/ToolBboxButton';
import { ToolBrushButton } from 'features/controlLayers/components/Tool/ToolBrushButton';
import { ToolColorPickerButton } from 'features/controlLayers/components/Tool/ToolColorPickerButton';
import { ToolMoveButton } from 'features/controlLayers/components/Tool/ToolMoveButton';
import { ToolRectButton } from 'features/controlLayers/components/Tool/ToolRectButton';
import { ToolTextButton } from 'features/controlLayers/components/Tool/ToolTextButton';
import React from 'react';

import { ToolEraserButton } from './ToolEraserButton';
import { ToolViewButton } from './ToolViewButton';

export const ToolChooser: React.FC = () => {
  return (
    <>
      <ButtonGroup isAttached orientation="vertical">
        <ToolBrushButton />
        <ToolEraserButton />
        <ToolRectButton />
        <ToolTextButton />
        <ToolMoveButton />
        <ToolViewButton />
        <ToolBboxButton />
        <ToolColorPickerButton />
      </ButtonGroup>
    </>
  );
};
