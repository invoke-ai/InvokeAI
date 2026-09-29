import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box } from '@invoke-ai/ui-library';
import { useMouseOverFormField, useMouseOverNode } from 'features/nodes/hooks/useMouseOverNode';
import { memo } from 'react';

const sx: SystemStyleObject = {
  position: 'absolute',
  top: 0,
  insetInlineEnd: 0,
  bottom: 0,
  insetInlineStart: 0,
  borderRadius: 'base',
  transitionProperty: 'none',
  pointerEvents: 'none',
  display: 'none',
  '&[data-is-mouse-over-node-or-form-field="true"]': {
    display: 'block',
    bg: 'invokeBlueAlpha.100',
  },
};

/**
 * Highlights a form element that references a node while the mouse is over that node (or vice versa), so it is easy
 * to see which node a form element belongs to. Must be rendered inside a relatively-positioned element.
 */
export const FormElementNodeOverlay = memo(({ nodeId }: { nodeId: string }) => {
  const mouseOverNode = useMouseOverNode(nodeId);
  const mouseOverFormField = useMouseOverFormField(nodeId);

  return (
    <Box
      sx={sx}
      data-is-mouse-over-node-or-form-field={mouseOverNode.isMouseOverNode || mouseOverFormField.isMouseOverFormField}
    />
  );
});
FormElementNodeOverlay.displayName = 'FormElementNodeOverlay';
