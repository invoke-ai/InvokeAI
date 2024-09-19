import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { INTERACTION_SCOPES } from 'common/hooks/interactionScopes';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsOutBold } from 'react-icons/pi';

export const CanvasToolbarResetViewButton = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const isCanvasActive = useStore(INTERACTION_SCOPES.canvas.$isActive);
  const imageViewer = useImageViewer();

  useRegisteredHotkeys({
    id: 'fitLayersToCanvas',
    category: 'canvas',
    callback: canvasManager.stage.fitLayersToStage,
    options: { enabled: isCanvasActive && !imageViewer.isOpen, preventDefault: true },
    dependencies: [isCanvasActive, imageViewer.isOpen],
  });
  useRegisteredHotkeys({
    id: 'setZoomTo100Percent',
    category: 'canvas',
    callback: () => canvasManager.stage.setScale(1),
    options: { enabled: isCanvasActive && !imageViewer.isOpen, preventDefault: true },
    dependencies: [isCanvasActive, imageViewer.isOpen],
  });
  useRegisteredHotkeys({
    id: 'setZoomTo200Percent',
    category: 'canvas',
    callback: () => canvasManager.stage.setScale(2),
    options: { enabled: isCanvasActive && !imageViewer.isOpen, preventDefault: true },
    dependencies: [isCanvasActive, imageViewer.isOpen],
  });
  useRegisteredHotkeys({
    id: 'setZoomTo400Percent',
    category: 'canvas',
    callback: () => canvasManager.stage.setScale(4),
    options: { enabled: isCanvasActive && !imageViewer.isOpen, preventDefault: true },
    dependencies: [isCanvasActive, imageViewer.isOpen],
  });
  useRegisteredHotkeys({
    id: 'setZoomTo800Percent',
    category: 'canvas',
    callback: () => canvasManager.stage.setScale(8),
    options: { enabled: isCanvasActive && !imageViewer.isOpen, preventDefault: true },
    dependencies: [isCanvasActive, imageViewer.isOpen],
  });

  return (
    <IconButton
      tooltip={t('hotkeys.canvas.fitLayersToCanvas.title')}
      aria-label={t('hotkeys.canvas.fitLayersToCanvas.title')}
      onClick={canvasManager.stage.fitLayersToStage}
      icon={<PiArrowsOutBold />}
      variant="link"
      alignSelf="stretch"
    />
  );
});

CanvasToolbarResetViewButton.displayName = 'CanvasToolbarResetViewButton';
