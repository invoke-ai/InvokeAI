/* eslint-disable i18next/no-literal-string */
import { Flex, Spacer } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasModeSwitcher } from 'features/controlLayers/components/CanvasModeSwitcher';
import { CanvasResetViewButton } from 'features/controlLayers/components/CanvasResetViewButton';
import { CanvasScale } from 'features/controlLayers/components/CanvasScale';
import { CanvasSettingsPopover } from 'features/controlLayers/components/Settings/CanvasSettingsPopover';
import { ToolBrushWidth } from 'features/controlLayers/components/Tool/ToolBrushWidth';
import { ToolChooser } from 'features/controlLayers/components/Tool/ToolChooser';
import { ToolEraserWidth } from 'features/controlLayers/components/Tool/ToolEraserWidth';
import { ToolFillColorPicker } from 'features/controlLayers/components/Tool/ToolFillColorPicker';
import { UndoRedoButtonGroup } from 'features/controlLayers/components/UndoRedoButtonGroup';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { ToggleProgressButton } from 'features/gallery/components/ImageViewer/ToggleProgressButton';
import { ViewerToggleMenu } from 'features/gallery/components/ImageViewer/ViewerToggleMenu';
import { memo } from 'react';

export const ControlLayersToolbar = memo(() => {
  const tool = useAppSelector((s) => s.canvasV2.tool.selected);
  return (
    <CanvasManagerProviderGate>
      <Flex w="full" gap={2} alignItems="center">
        <ToggleProgressButton />
        <ToolChooser />
        <Spacer />
        {tool === 'brush' && <ToolBrushWidth />}
        {tool === 'eraser' && <ToolEraserWidth />}
        <Spacer />
        <CanvasScale />
        <CanvasResetViewButton />
        <Spacer />
        <ToolFillColorPicker />
        <CanvasModeSwitcher />
        <UndoRedoButtonGroup />
        <CanvasSettingsPopover />
        <ViewerToggleMenu />
      </Flex>
    </CanvasManagerProviderGate>
  );
});

ControlLayersToolbar.displayName = 'ControlLayersToolbar';
