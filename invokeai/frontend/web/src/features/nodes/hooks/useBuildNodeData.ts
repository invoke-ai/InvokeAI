import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { useCallback } from 'react';
import { Node, useReactFlow } from 'reactflow';
import { AnyInvocationType } from 'services/events/types';
import { buildNodeData } from '../store/util/buildNodeData';
import { DRAG_HANDLE_CLASSNAME, NODE_WIDTH } from '../types/constants';

const templatesSelector = createSelector(
  [(state: RootState) => state.nodes],
  (nodes) => nodes.nodeTemplates
);

export const SHARED_NODE_PROPERTIES: Partial<Node> = {
  dragHandle: `.${DRAG_HANDLE_CLASSNAME}`,
};

export const useBuildNodeData = () => {
  const nodeTemplates = useAppSelector(templatesSelector);

  const flow = useReactFlow();

  return useCallback(
    (type: AnyInvocationType | 'current_image' | 'notes') => {
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

      const template = nodeTemplates[type];

      return buildNodeData(type, position, template);
    },
    [nodeTemplates, flow]
  );
};
