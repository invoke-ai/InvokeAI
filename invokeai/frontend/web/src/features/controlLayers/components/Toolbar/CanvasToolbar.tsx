import { Divider, Flex } from '@invoke-ai/ui-library';
import useResizeObserver from '@react-hook/resize-observer';
import { CanvasSettingsPopover } from 'features/controlLayers/components/Settings/CanvasSettingsPopover';
import { useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { ToolFillColorPicker } from 'features/controlLayers/components/Tool/ToolFillColorPicker';
import { ToolWidthPicker } from 'features/controlLayers/components/Tool/ToolWidthPicker';
import { CanvasToolbarFitBboxToLayersButton } from 'features/controlLayers/components/Toolbar/CanvasToolbarFitBboxToLayersButton';
import { CanvasToolbarFitBboxToMasksButton } from 'features/controlLayers/components/Toolbar/CanvasToolbarFitBboxToMasksButton';
import { CanvasToolbarNewSessionMenuButton } from 'features/controlLayers/components/Toolbar/CanvasToolbarNewSessionMenuButton';
import { CanvasToolbarRedoButton } from 'features/controlLayers/components/Toolbar/CanvasToolbarRedoButton';
import { CanvasToolbarResetViewButton } from 'features/controlLayers/components/Toolbar/CanvasToolbarResetViewButton';
import { CanvasToolbarSaveToGalleryButton } from 'features/controlLayers/components/Toolbar/CanvasToolbarSaveToGalleryButton';
import { CanvasToolbarScale } from 'features/controlLayers/components/Toolbar/CanvasToolbarScale';
import { CanvasToolbarUndoButton } from 'features/controlLayers/components/Toolbar/CanvasToolbarUndoButton';
import { useCanvasDeleteLayerHotkey } from 'features/controlLayers/hooks/useCanvasDeleteLayerHotkey';
import { useCanvasEntityQuickSwitchHotkey } from 'features/controlLayers/hooks/useCanvasEntityQuickSwitchHotkey';
import { useCanvasFilterHotkey } from 'features/controlLayers/hooks/useCanvasFilterHotkey';
import { useCanvasInvertMaskHotkey } from 'features/controlLayers/hooks/useCanvasInvertMaskHotkey';
import { useCanvasResetLayerHotkey } from 'features/controlLayers/hooks/useCanvasResetLayerHotkey';
import { useCanvasToggleBboxHotkey } from 'features/controlLayers/hooks/useCanvasToggleBboxHotkey';
import { useCanvasToggleNonRasterLayersHotkey } from 'features/controlLayers/hooks/useCanvasToggleNonRasterLayersHotkey';
import { useCanvasTransformHotkey } from 'features/controlLayers/hooks/useCanvasTransformHotkey';
import { useCanvasUndoRedoHotkeys } from 'features/controlLayers/hooks/useCanvasUndoRedoHotkeys';
import { useNextPrevEntityHotkeys } from 'features/controlLayers/hooks/useNextPrevEntity';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';

export const CanvasToolbar = memo(() => {
  const toolbarRef = useRef(null);
  const [toolBarWidth, setToolBarWidth] = useState(0);
  const [fillColorPickerWidth, setFillColorPickerWidth] = useState(0);
  const [toolWidthPickerWidth, setToolWidthPickerWidth] = useState(0);

  const isBrushSelected = useToolIsSelected('brush');
  const isEraserSelected = useToolIsSelected('eraser');
  const showToolWithPicker = useMemo(() => {
    return isBrushSelected || isEraserSelected;
  }, [isBrushSelected, isEraserSelected]);

  const toolBarAvailableSpace = useMemo(() => {
    return toolBarWidth - fillColorPickerWidth - toolWidthPickerWidth;
  }, [toolBarWidth, fillColorPickerWidth, toolWidthPickerWidth]);

  useResizeObserver(toolbarRef, (entry) => setToolBarWidth(entry.contentRect.width));

  const onFillColorPickerWidthChange = useCallback(
    (width: number) => {
      setFillColorPickerWidth(width);
    },
    [setFillColorPickerWidth]
  );

  const onToolWidthPickerWidthChange = useCallback(
    (width: number) => {
      setToolWidthPickerWidth(width);
    },
    [setToolWidthPickerWidth]
  );

  useEffect(() => {
    if (!showToolWithPicker) {
      setToolWidthPickerWidth(0);
    }
  }, [showToolWithPicker]);

  useCanvasResetLayerHotkey();
  useCanvasDeleteLayerHotkey();
  useCanvasUndoRedoHotkeys();
  useCanvasEntityQuickSwitchHotkey();
  useNextPrevEntityHotkeys();
  useCanvasTransformHotkey();
  useCanvasFilterHotkey();
  useCanvasInvertMaskHotkey();
  useCanvasToggleNonRasterLayersHotkey();
  useCanvasToggleBboxHotkey();

  return (
    <Flex w="full" gap={2} alignItems="center" px={2}>
      <Flex ref={toolbarRef} w="full">
        <ToolFillColorPicker onComponentWidthChange={onFillColorPickerWidthChange} />
        {showToolWithPicker && (
          <ToolWidthPicker
            toolBarAvailableSpace={toolBarAvailableSpace}
            onComponentWidthChange={onToolWidthPickerWidthChange}
          />
        )}
      </Flex>
      <Flex alignItems="center" h="full" flexGrow={1} justifyContent="flex-end">
        <CanvasToolbarScale />
        <CanvasToolbarResetViewButton />
        <CanvasToolbarFitBboxToLayersButton />
        <CanvasToolbarFitBboxToMasksButton />
      </Flex>
      <Divider orientation="vertical" />
      <Flex alignItems="center" h="full">
        <CanvasToolbarSaveToGalleryButton />
        <CanvasToolbarUndoButton />
        <CanvasToolbarRedoButton />
        <CanvasToolbarNewSessionMenuButton />
        <CanvasSettingsPopover />
      </Flex>
    </Flex>
  );
});

CanvasToolbar.displayName = 'CanvasToolbar';
