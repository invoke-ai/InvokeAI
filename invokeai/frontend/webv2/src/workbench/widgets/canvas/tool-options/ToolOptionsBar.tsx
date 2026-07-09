import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { ToolId } from '@workbench/canvas-engine/types';
import type { ComponentType } from 'react';

import { HStack, Text } from '@chakra-ui/react';
import { CanvasFloatingBarDivider } from '@workbench/widgets/canvas/CanvasFloatingBar';
import { useCanvasActiveTool, useCanvasZoom } from '@workbench/widgets/canvas/engineStoreHooks';
import { formatZoomPercent } from '@workbench/widgets/canvas/zoomOptions';

import { BboxOptions } from './BboxOptions';
import { BrushOptions } from './BrushOptions';
import { CanvasOptionsBar } from './CanvasOptionsBar';
import { EraserOptions } from './EraserOptions';
import { GradientOptions } from './GradientOptions';
import { LassoOptions } from './LassoOptions';
import { MoveOptions } from './MoveOptions';
import { ShapeOptions } from './ShapeOptions';
import { TextOptions } from './TextOptions';
import { TransformOptions } from './TransformOptions';

/** Props every per-tool options component receives — just the shared engine handle. */
export interface ToolOptionsComponentProps {
  engine: CanvasEngine;
}

/**
 * Contextual options content per active tool. Tools without an entry here
 * (view, and anything not yet implemented) render no controls — the bar still
 * shows the document info on the right.
 */
export const TOOL_OPTIONS_COMPONENTS: Partial<Record<ToolId, ComponentType<ToolOptionsComponentProps>>> = {
  bbox: BboxOptions,
  brush: BrushOptions,
  eraser: EraserOptions,
  gradient: GradientOptions,
  lasso: LassoOptions,
  move: MoveOptions,
  shape: ShapeOptions,
  text: TextOptions,
  transform: TransformOptions,
};

/**
 * The canvas's floating tool-options bar (bottom-center over the surface):
 * contextual controls for the active tool on the left, then the document
 * dimensions / zoom read-out on the right (absorbed from the former floating HUD).
 * Tool options read and write the engine's
 * transient option stores directly (`useBrushOptions` / `useEraserOptions` +
 * `engine.stores.*.set(...)`) — there is no React state mirror. Positioned by
 * {@link CanvasWidgetView}; shares its look with the staging bar via
 * {@link CanvasFloatingBar}.
 */
export const ToolOptionsBar = ({
  documentHeight,
  documentWidth,
  engine,
}: {
  documentHeight: number | null;
  documentWidth: number | null;
  engine: CanvasEngine;
}) => {
  const activeTool = useCanvasActiveTool(engine);
  const zoom = useCanvasZoom(engine);
  const OptionsComponent = TOOL_OPTIONS_COMPONENTS[activeTool];
  const hasDocument = documentWidth !== null && documentHeight !== null;

  return (
    <CanvasOptionsBar>
      {OptionsComponent ? (
        <HStack align="center" gap="3" minW="0" overflow="hidden">
          <OptionsComponent engine={engine} />
        </HStack>
      ) : null}
      {OptionsComponent && hasDocument ? <CanvasFloatingBarDivider /> : null}
      {hasDocument ? (
        <HStack
          align="center"
          color="fg.muted"
          flexShrink="0"
          fontSize="2xs"
          fontVariantNumeric="tabular-nums"
          gap="2"
          px="1"
        >
          <Text>
            {documentWidth} × {documentHeight} @ {formatZoomPercent(zoom)}
          </Text>
        </HStack>
      ) : null}
    </CanvasOptionsBar>
  );
};
