import type { ToolId } from '@workbench/canvas-engine/api';
import type { CanvasOperationState } from '@workbench/canvas-operations/api';
import type { CanvasEngineHandle } from '@workbench/widgets/canvas/useCanvasEngine';
import type { ComponentType } from 'react';

import { HStack } from '@chakra-ui/react';
import { BboxDetailsBar } from '@workbench/widgets/canvas/BboxDetailsBar';
import { CanvasFloatingBarDivider } from '@workbench/widgets/canvas/CanvasFloatingBar';
import { useCanvasActiveTool, useCanvasOperation } from '@workbench/widgets/canvas/engineStoreHooks';

import { BboxOptions } from './BboxOptions';
import { BrushOptions } from './BrushOptions';
import { CanvasOperationBar } from './CanvasOperationBar';
import { CanvasOptionsBar } from './CanvasOptionsBar';
import { EraserOptions } from './EraserOptions';
import { GradientOptions } from './GradientOptions';
import { LassoOptions } from './LassoOptions';
import { MarqueeOptions } from './MarqueeOptions';
import { MoveOptions } from './MoveOptions';
import { ShapeOptions } from './ShapeOptions';
import { TextOptions } from './TextOptions';
import { TransformOptions } from './TransformOptions';

export type CanvasToolOptionsEngine = Pick<
  CanvasEngineHandle,
  'document' | 'interaction' | 'layers' | 'projectId' | 'selection' | 'tools' | 'viewport'
>;

/** Props every per-tool options component receives — just the shared engine handle. */
export interface ToolOptionsComponentProps {
  engine: CanvasToolOptionsEngine;
}

/**
 * Contextual options content per active tool. Tools without an entry here
 * (view, and anything not yet implemented) render no controls, and the bar
 * itself is omitted.
 */
export const TOOL_OPTIONS_COMPONENTS: Partial<Record<ToolId, ComponentType<ToolOptionsComponentProps>>> = {
  bbox: BboxOptions,
  brush: BrushOptions,
  eraser: EraserOptions,
  gradient: GradientOptions,
  lasso: LassoOptions,
  marquee: MarqueeOptions,
  move: MoveOptions,
  shape: ShapeOptions,
  text: TextOptions,
  transform: TransformOptions,
};

export const resolveCanvasOptionsContent = (
  operation: Pick<CanvasOperationState, 'status'>,
  activeTool: ToolId
): 'operation' | ToolId | null => {
  if (operation.status === 'active') {
    return 'operation';
  }
  return TOOL_OPTIONS_COMPONENTS[activeTool] ? activeTool : null;
};

/**
 * The canvas's floating tool-options bar (bottom-center over the surface):
 * contextual controls for the active tool. Tool options read and write the
 * engine's transient option stores directly (`useBrushOptions` /
 * `useEraserOptions` + `engine.interaction.set(...)`) — there is no React state
 * mirror. Positioned by {@link CanvasWidgetView}; shares its look with the
 * staging bar via {@link CanvasFloatingBar}.
 *
 * The bar is purely contextual: a tool with no options renders nothing at all,
 * rather than an empty bar floating over the surface.
 */
export const ToolOptionsBar = ({ engine }: { engine: CanvasToolOptionsEngine }) => {
  const activeTool = useCanvasActiveTool(engine);
  const operation = useCanvasOperation(engine);
  const content = resolveCanvasOptionsContent(operation, activeTool);
  if (content === 'operation' && operation.status === 'active') {
    return <CanvasOperationBar engine={engine} isExternalInteractionLocked={false} operation={operation} />;
  }
  const OptionsComponent = content && content !== 'operation' ? TOOL_OPTIONS_COMPONENTS[content] : undefined;
  const hasBboxDetails = activeTool === 'bbox';

  if (!OptionsComponent && !hasBboxDetails) {
    return null;
  }

  return (
    <CanvasOptionsBar>
      <HStack align="center" gap="3" minW="0" overflow="hidden">
        {hasBboxDetails ? <BboxDetailsBar engine={engine} /> : null}
        {hasBboxDetails && OptionsComponent ? <CanvasFloatingBarDivider /> : null}
        {OptionsComponent ? <OptionsComponent engine={engine} /> : null}
      </HStack>
    </CanvasOptionsBar>
  );
};
