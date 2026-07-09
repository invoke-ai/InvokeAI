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

import { useCanvasActiveTool } from './engineStoreHooks';

interface ToolStripButtonProps {
  engine: CanvasEngine;
  icon: typeof HandIcon;
  label: string;
  toolId: ToolId;
}

/** One sticky tool button: active state comes straight from the engine's transient store, click drives `engine.setTool`. */
const ToolStripButton = ({ engine, icon, label, toolId }: ToolStripButtonProps) => {
  const activeTool = useCanvasActiveTool(engine);
  const onClick = useCallback(() => engine.setTool(toolId), [engine, toolId]);

  return <ToolbarButton icon={icon} isActive={activeTool === toolId} label={label} onClick={onClick} />;
};

/**
 * The canvas's left-docked, vertical tool strip. Color-picker is intentionally
 * absent — it's alt-hold-only for now (see `canvas-engine/input/pointerPipeline.ts`),
 * not a sticky tool a user selects directly.
 */
const ToolStripRoot = ({ engine }: { engine: CanvasEngine }) => {
  const { t } = useTranslation();

  return (
    <Box left="2" position="absolute" top="50%" transform="translateY(-50%)" zIndex="2">
      <Toolbar aria-label={t('widgets.canvas.tools.label')}>
        <ToolStripButton engine={engine} icon={HandIcon} label={t('widgets.canvas.tools.view')} toolId="view" />
        <ToolStripButton engine={engine} icon={MoveIcon} label={t('widgets.canvas.tools.move')} toolId="move" />
        <ToolStripButton
          engine={engine}
          icon={Rotate3dIcon}
          label={t('widgets.canvas.tools.transform')}
          toolId="transform"
        />
        <ToolStripButton engine={engine} icon={FrameIcon} label={t('widgets.canvas.tools.bbox')} toolId="bbox" />
        <ToolStripButton engine={engine} icon={BrushIcon} label={t('widgets.canvas.tools.brush')} toolId="brush" />
        <ToolStripButton engine={engine} icon={EraserIcon} label={t('widgets.canvas.tools.eraser')} toolId="eraser" />
        <ToolStripButton engine={engine} icon={SquareIcon} label={t('widgets.canvas.tools.shape')} toolId="shape" />
        <ToolStripButton
          engine={engine}
          icon={PaintBucketIcon}
          label={t('widgets.canvas.tools.gradient')}
          toolId="gradient"
        />
        <ToolStripButton engine={engine} icon={TypeIcon} label={t('widgets.canvas.tools.text')} toolId="text" />
        <ToolStripButton engine={engine} icon={LassoIcon} label={t('widgets.canvas.tools.lasso')} toolId="lasso" />
      </Toolbar>
    </Box>
  );
};

export const ToolStrip = Object.assign(ToolStripRoot, { Button: ToolStripButton });
