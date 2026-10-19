import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';

import { useEffect } from 'react';

import { useActiveColorCommands, useActiveColorPair } from './useActiveColors';

/**
 * Isolate frequent color-pair updates in this null adapter. Mirror the pair into brush color, captured at stroke
 * start, and route unclaimed samples to the active pair target.
 */
export const CanvasColorFeed = ({ engine }: { engine: CanvasEngineHandle | null }) => {
  const colorPair = useActiveColorPair();
  const colorCommands = useActiveColorCommands();

  useEffect(() => {
    if (!engine) {
      return;
    }
    const options = engine.interaction.get('brushOptions');
    if (options.color !== colorPair.foreground) {
      engine.interaction.set('brushOptions', { ...options, color: colorPair.foreground });
    }
  }, [engine, colorPair.foreground]);

  // The whole pair, for gesture-start reads: new shapes and text sessions
  // capture it when they begin, and the gradient FG→BG preset resolves it.
  useEffect(() => {
    if (!engine) {
      return;
    }
    engine.interaction.set('colorPair', { background: colorPair.background, foreground: colorPair.foreground });
  }, [engine, colorPair.background, colorPair.foreground]);

  useEffect(() => {
    if (!engine) {
      return;
    }
    return engine.tools.setColorSampleRouter((hex) => {
      colorCommands.applySampledColor(hex);
      return true;
    });
  }, [colorCommands, engine]);

  return null;
};
