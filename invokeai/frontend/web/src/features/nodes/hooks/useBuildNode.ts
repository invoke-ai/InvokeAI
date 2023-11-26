import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { useCallback } from 'react';
import { Node, useReactFlow } from 'reactflow';
import { DRAG_HANDLE_CLASSNAME, NODE_WIDTH } from '../types/constants';
import { AnyNode, InvocationTemplate } from '../types/invocation';
import { buildCurrentImageNode } from '../util/node/buildCurrentImageNode';
import { buildInvocationNode } from '../util/node/buildInvocationNode';
import { buildNotesNode } from '../util/node/buildNotesNode';

const templatesSelector = createSelector(
  [(state: RootState) => state.nodes],
  (nodes) => nodes.nodeTemplates
);

export const SHARED_NODE_PROPERTIES: Partial<Node> = {
  dragHandle: `.${DRAG_HANDLE_CLASSNAME}`,
};

export const useBuildNode = () => {
  const nodeTemplates = useAppSelector(templatesSelector);

  const flow = useReactFlow();

  return useCallback(
    // string here is "any invocation type"
    (type: string | 'current_image' | 'notes'): AnyNode => {
      let _x = window.innerWidth / 2;
      let _y = window.innerHeight / 2;

      // attempt to center the node in the middle of the flow
      const rect = document
        .querySelector('#workflow-editor')
        ?.getBoundingClientRect();

      if (rect) {
        _x = rect.width / 2 - NODE_WIDTH / 2;
        _y = rect.height / 2 - NODE_WIDTH / 2;
      }

      const position = flow.project({
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
      const template = nodeTemplates[type] as InvocationTemplate;

      return buildInvocationNode(position, template);
    },
    [nodeTemplates, flow]
  );
};
