import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { canvasInstanceAdded } from 'features/controlLayers/store/canvasesSlice';
import { selectCanvasCount } from 'features/controlLayers/store/selectors';
import type { DockviewPanelParameters } from 'features/ui/layouts/auto-layout-context';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { 
  DOCKVIEW_TAB_CANVAS_WORKSPACE_ID, 
  WORKSPACE_PANEL_ID 
} from 'features/ui/layouts/shared';
import { nanoid } from 'nanoid';
import { memo, useCallback } from 'react';
import { PiPlus } from 'react-icons/pi';

interface CanvasInstanceManagerProps {
  maxCanvases?: number;
}

export const CanvasInstanceManager = memo(({ maxCanvases = 3 }: CanvasInstanceManagerProps) => {
  const dispatch = useAppDispatch();
  const canvasCount = useAppSelector(selectCanvasCount);
  
  const addCanvas = useCallback(() => {
    if (canvasCount >= maxCanvases) {
      return;
    }
    
    const canvasId = nanoid();
    const canvasName = `Canvas ${canvasCount + 1}`;
    
    // Add to Redux first
    dispatch(canvasInstanceAdded({ canvasId, name: canvasName }));
    
    // Get the dockview API and create the panel
    const dockviewApi = navigationApi.getDockviewApi('canvas', 'main');
    if (dockviewApi) {
      const panelId = `${WORKSPACE_PANEL_ID}_${canvasId}`;
      
      // Create the dockview panel
      dockviewApi.addPanel<DockviewPanelParameters>({
        id: panelId,
        component: WORKSPACE_PANEL_ID,
        title: canvasName,
        tabComponent: DOCKVIEW_TAB_CANVAS_WORKSPACE_ID,
        params: {
          tab: 'canvas',
          canvasId,
          focusRegion: 'canvas',
          i18nKey: 'ui.panels.canvas',
        },
      });
      
      // Activate the new panel
      const newPanel = dockviewApi.getPanel(panelId);
      newPanel?.api.setActive();
    }
  }, [canvasCount, maxCanvases, dispatch]);
  
  const canCanAddCanvas = canvasCount < maxCanvases;
  
  if (canvasCount === 0) {
    return null;
  }
  
  return (
    <Flex gap={2} alignItems="center">
      <Text fontSize="sm" color="base.300">
        Canvases: {canvasCount}/{maxCanvases}
      </Text>
      
      {canCanAddCanvas && (
        <IconButton
          aria-label="Add Canvas"
          icon={<PiPlus />}
          size="sm"
          variant="ghost"
          onClick={addCanvas}
          tooltip="Add new canvas"
        />
      )}
    </Flex>
  );
});

CanvasInstanceManager.displayName = 'CanvasInstanceManager';