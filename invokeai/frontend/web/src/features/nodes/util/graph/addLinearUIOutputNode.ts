import type { RootState } from 'app/store/store';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import type { LinearUIOutputInvocation, NonNullableGraph } from 'services/api/types';

import {
  CANVAS_OUTPUT,
  LATENTS_TO_IMAGE,
  LATENTS_TO_IMAGE_HRF_HR,
  LINEAR_UI_OUTPUT,
  NSFW_CHECKER,
  WATERMARKER,
} from './constants';

/**
 * Set the `use_cache` field on the linear/canvas graph's final image output node to False.
 */
export const addLinearUIOutputNode = (state: RootState, graph: NonNullableGraph): void => {
  const activeTabName = activeTabNameSelector(state);
  const is_intermediate = activeTabName === 'unifiedCanvas' ? !state.canvas.shouldAutoSave : false;
  const { autoAddBoardId } = state.gallery;

  const linearUIOutputNode: LinearUIOutputInvocation = {
    id: LINEAR_UI_OUTPUT,
    type: 'linear_ui_output',
    is_intermediate,
    use_cache: false,
    board: autoAddBoardId === 'none' ? undefined : { board_id: autoAddBoardId },
  };

  graph.nodes[LINEAR_UI_OUTPUT] = linearUIOutputNode;

  const destination = {
    node_id: LINEAR_UI_OUTPUT,
    field: 'image',
  };

  if (WATERMARKER in graph.nodes) {
    graph.edges.push({
      source: {
        node_id: WATERMARKER,
        field: 'image',
      },
      destination,
    });
  } else if (NSFW_CHECKER in graph.nodes) {
    graph.edges.push({
      source: {
        node_id: NSFW_CHECKER,
        field: 'image',
      },
      destination,
    });
  } else if (CANVAS_OUTPUT in graph.nodes) {
    graph.edges.push({
      source: {
        node_id: CANVAS_OUTPUT,
        field: 'image',
      },
      destination,
    });
  } else if (LATENTS_TO_IMAGE_HRF_HR in graph.nodes) {
    graph.edges.push({
      source: {
        node_id: LATENTS_TO_IMAGE_HRF_HR,
        field: 'image',
      },
      destination,
    });
  } else if (LATENTS_TO_IMAGE in graph.nodes) {
    graph.edges.push({
      source: {
        node_id: LATENTS_TO_IMAGE,
        field: 'image',
      },
      destination,
    });
  }
};
