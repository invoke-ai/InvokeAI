import type { SelectionOp } from '@workbench/canvas-engine/api';

import { useLassoOptions } from '@workbench/widgets/canvas/engineStoreHooks';
import { useCallback } from 'react';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

import { SelectionOptionsRow } from './SelectionOptionsRow';

/**
 * Lasso tool options: the shared selection controls, bound to the lasso's own
 * persistent op mode.
 */
export const LassoOptions = ({ engine }: ToolOptionsComponentProps) => {
  const options = useLassoOptions(engine);

  const onModeChange = useCallback(
    (mode: SelectionOp) => engine.interaction.set('lassoOptions', { ...options, mode }),
    [engine, options]
  );

  return (
    <SelectionOptionsRow
      engine={engine}
      hintKey="widgets.canvas.toolOptions.lassoHint"
      mode={options.mode}
      onModeChange={onModeChange}
    />
  );
};
