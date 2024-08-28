/* eslint-disable i18next/no-literal-string */
import { Flex, Spacer } from '@invoke-ai/ui-library';
import { CanvasModeSwitcher } from 'features/controlLayers/components/CanvasModeSwitcher';
import { CanvasResetViewButton } from 'features/controlLayers/components/CanvasResetViewButton';
import { CanvasScale } from 'features/controlLayers/components/CanvasScale';
import { CanvasSettingsPopover } from 'features/controlLayers/components/Settings/CanvasSettingsPopover';
import { ToolChooser } from 'features/controlLayers/components/Tool/ToolChooser';
import { ToolFillColorPicker } from 'features/controlLayers/components/Tool/ToolFillColorPicker';
import { ToolSettings } from 'features/controlLayers/components/Tool/ToolSettings';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCanvasUndoRedo } from 'features/controlLayers/hooks/useCanvasUndoRedo';
import { ToggleProgressButton } from 'features/gallery/components/ImageViewer/ToggleProgressButton';
import { ViewerToggleMenu } from 'features/gallery/components/ImageViewer/ViewerToggleMenu';
import { memo } from 'react';

export const ControlLayersToolbar = memo(() => {
  useCanvasUndoRedo();

  return (
    <CanvasManagerProviderGate>
      <Flex w="full" gap={2} alignItems="center">
        <ToggleProgressButton />
        <ToolChooser />
        <Spacer />
        <ToolSettings />
        <Spacer />
        <CanvasScale />
        <CanvasResetViewButton />
        <Spacer />
        <ToolFillColorPicker />
        <CanvasModeSwitcher />
        <CanvasSettingsPopover />
        <ViewerToggleMenu />
      </Flex>
    </CanvasManagerProviderGate>
  );
});

ControlLayersToolbar.displayName = 'ControlLayersToolbar';
