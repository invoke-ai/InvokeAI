import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { ToolId } from '@workbench/canvas-engine/types';

import { Box } from '@chakra-ui/react';
import { Toolbar, ToolbarButton } from '@workbench/components/ui';
import {
  BrushIcon,
  EraserIcon,
  FrameIcon,
  HandIcon,
  LassoIcon,
  MoveIcon,
  PaintBucketIcon,
  Rotate3dIcon,
  SquareIcon,
  TypeIcon,
} from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { isCanvasToolEnabled } from './canvasInteractionLock';
import { useCanvasActiveTool } from './engineStoreHooks';

interface ToolStripButtonProps {
  engine: CanvasEngine;
  icon: typeof HandIcon;
  isInteractionLocked: boolean;
  label: string;
  toolId: ToolId;
}

/** One sticky tool button: active state comes straight from the engine's transient store, click drives `engine.setTool`. */
const ToolStripButton = ({ engine, icon, isInteractionLocked, label, toolId }: ToolStripButtonProps) => {
  const activeTool = useCanvasActiveTool(engine);
  const isDisabled = !isCanvasToolEnabled(toolId, isInteractionLocked);
  const onClick = useCallback(() => engine.setTool(toolId), [engine, toolId]);

  return (
    <ToolbarButton disabled={isDisabled} icon={icon} isActive={activeTool === toolId} label={label} onClick={onClick} />
  );
};

/**
 * The canvas's left-docked, vertical tool strip. Color-picker is intentionally
 * absent — it's alt-hold-only for now (see `canvas-engine/input/pointerPipeline.ts`),
 * not a sticky tool a user selects directly.
 */
const ToolStripRoot = ({
  engine,
  isInteractionLocked = false,
}: {
  engine: CanvasEngine;
  isInteractionLocked?: boolean;
}) => {
  const { t } = useTranslation();

  return (
    <Box left="2" position="absolute" top="50%" transform="translateY(-50%)" zIndex="2">
      <Toolbar aria-label={t('widgets.canvas.tools.label')}>
        <ToolStripButton
          engine={engine}
          icon={HandIcon}
          isInteractionLocked={isInteractionLocked}
          label={t('widgets.canvas.tools.view')}
          toolId="view"
        />
        <ToolStripButton
          engine={engine}
          icon={MoveIcon}
          isInteractionLocked={isInteractionLocked}
          label={t('widgets.canvas.tools.move')}
          toolId="move"
        />
        <ToolStripButton
          engine={engine}
          icon={Rotate3dIcon}
          isInteractionLocked={isInteractionLocked}
          label={t('widgets.canvas.tools.transform')}
          toolId="transform"
        />
        <ToolStripButton
          engine={engine}
          icon={FrameIcon}
          isInteractionLocked={isInteractionLocked}
          label={t('widgets.canvas.tools.bbox')}
          toolId="bbox"
        />
        <ToolStripButton
          engine={engine}
          icon={BrushIcon}
          isInteractionLocked={isInteractionLocked}
          label={t('widgets.canvas.tools.brush')}
          toolId="brush"
        />
        <ToolStripButton
          engine={engine}
          icon={EraserIcon}
          isInteractionLocked={isInteractionLocked}
          label={t('widgets.canvas.tools.eraser')}
          toolId="eraser"
        />
        <ToolStripButton
          engine={engine}
          icon={SquareIcon}
          isInteractionLocked={isInteractionLocked}
          label={t('widgets.canvas.tools.shape')}
          toolId="shape"
        />
        <ToolStripButton
          engine={engine}
          icon={PaintBucketIcon}
          isInteractionLocked={isInteractionLocked}
          label={t('widgets.canvas.tools.gradient')}
          toolId="gradient"
        />
        <ToolStripButton
          engine={engine}
          icon={TypeIcon}
          isInteractionLocked={isInteractionLocked}
          label={t('widgets.canvas.tools.text')}
          toolId="text"
        />
        <ToolStripButton
          engine={engine}
          icon={LassoIcon}
          isInteractionLocked={isInteractionLocked}
          label={t('widgets.canvas.tools.lasso')}
          toolId="lasso"
        />
      </Toolbar>
    </Box>
  );
};

export const ToolStrip = Object.assign(ToolStripRoot, { Button: ToolStripButton });
