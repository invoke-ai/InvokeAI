import { ButtonGroup } from '@invoke-ai/ui-library';
import { ToolBboxButton } from 'features/controlLayers/components/Tool/ToolBboxButton';
import { ToolBrushButton } from 'features/controlLayers/components/Tool/ToolBrushButton';
import { ToolColorPickerButton } from 'features/controlLayers/components/Tool/ToolColorPickerButton';
import { ToolGradientButton } from 'features/controlLayers/components/Tool/ToolGradientButton';
import { ToolMoveButton } from 'features/controlLayers/components/Tool/ToolMoveButton';
import { ToolRectButton } from 'features/controlLayers/components/Tool/ToolRectButton';
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
        <ToolGradientButton />
        <ToolMoveButton />
        <ToolViewButton />
        <ToolBboxButton />
        <ToolColorPickerButton />
      </ButtonGroup>
    </>
  );
};
