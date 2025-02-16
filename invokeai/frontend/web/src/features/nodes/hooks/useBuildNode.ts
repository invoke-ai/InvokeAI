import { useStore } from '@nanostores/react';
import { useReactFlow } from '@xyflow/react';
import { $templates } from 'features/nodes/store/nodesSlice';
import { NODE_WIDTH } from 'features/nodes/types/constants';
import type { AnyNode, InvocationTemplate } from 'features/nodes/types/invocation';
import { buildCurrentImageNode } from 'features/nodes/util/node/buildCurrentImageNode';
import { buildInvocationNode } from 'features/nodes/util/node/buildInvocationNode';
import { buildNotesNode } from 'features/nodes/util/node/buildNotesNode';
import { useCallback } from 'react';

export const useBuildNode = () => {
  const templates = useStore($templates);
  const { screenToFlowPosition } = useReactFlow();

  return useCallback(
    // string here is "any invocation type"
    (type: string | 'current_image' | 'notes'): AnyNode => {
      let _x = window.innerWidth / 2;
      let _y = window.innerHeight / 2;

      // attempt to center the node in the middle of the flow
      const rect = document.querySelector('#workflow-editor')?.getBoundingClientRect();

      if (rect) {
        _x = rect.width / 2 - NODE_WIDTH / 2 + rect.left;
        _y = rect.height / 2 - NODE_WIDTH / 2 + rect.top;
      }

      const position = screenToFlowPosition({
        x: _x,
        y: _y,
      });

      if (type === 'current_image') {
        return buildCurrentImageNode(position);
      }

      if (type === 'notes') {
        return buildNotesNode(position);
      }

      // TODO: Keep track of invocation types so we do not need to cast this
      // We know it is safe because the caller of this function gets the `type` arg from the list of invocation templates.
      const template = templates[type] as InvocationTemplate;

      return buildInvocationNode(position, template);
    },
    [screenToFlowPosition, templates]
  );
};
