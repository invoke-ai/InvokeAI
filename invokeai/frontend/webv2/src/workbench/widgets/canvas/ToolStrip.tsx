import type { CanvasCoreStoreCapability, CanvasToolCapability, ToolId } from '@workbench/canvas-engine/api';

import { Box } from '@chakra-ui/react';
import { Toolbar, ToolbarButton } from '@platform/ui';
import {
  BrushIcon,
  EraserIcon,
  FrameIcon,
  HandIcon,
  LassoIcon,
  MoveIcon,
  PaintBucketIcon,
  Rotate3dIcon,
  ShapesIcon,
  SquareDashedIcon,
  TypeIcon,
} from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { isCanvasToolEnabled } from './canvasInteractionLock';
import { useCanvasActiveTool } from './engineStoreHooks';

type ToolStripEngine = CanvasCoreStoreCapability & { readonly tools: CanvasToolCapability };

/**
 * Clears the center region's floating chrome, whose inset already includes the
 * gap below the islands. Outside the center the variable is undefined and the
 * strip falls back to its own top margin.
 */
const TOOL_STRIP_TOP = 'var(--wb-center-chrome-inset, var(--chakra-spacing-2))';

interface ToolStripButtonProps {
  engine: ToolStripEngine;
  icon: typeof HandIcon;
  isInteractionLocked: boolean;
  label: string;
  toolId: ToolId;
}

/** One sticky tool button: active state comes from core stores; click drives the tool capability. */
const ToolStripButton = ({ engine, icon, isInteractionLocked, label, toolId }: ToolStripButtonProps) => {
  const activeTool = useCanvasActiveTool(engine);
  const isDisabled = !isCanvasToolEnabled(toolId, isInteractionLocked);
  const onClick = useCallback(() => engine.tools.setTool(toolId), [engine, toolId]);

  return (
    <ToolbarButton disabled={isDisabled} icon={icon} isActive={activeTool === toolId} label={label} onClick={onClick} />
  );
};

/**
 * The canvas's left-docked, vertical tool strip, topped out directly beneath
 * the region's floating chrome islands. Color-picker is intentionally absent —
 * it's alt-hold-only for now (see `canvas-engine/input/pointerPipeline.ts`),
 * not a sticky tool a user selects directly.
 */
const ToolStripRoot = ({
  engine,
  isInteractionLocked = false,
}: {
  engine: ToolStripEngine;
  isInteractionLocked?: boolean;
}) => {
  const { t } = useTranslation();

  return (
    <Box left="2" position="absolute" top={TOOL_STRIP_TOP} zIndex="2">
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
          icon={ShapesIcon}
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
          icon={SquareDashedIcon}
          isInteractionLocked={isInteractionLocked}
          label={t('widgets.canvas.tools.marquee')}
          toolId="marquee"
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
