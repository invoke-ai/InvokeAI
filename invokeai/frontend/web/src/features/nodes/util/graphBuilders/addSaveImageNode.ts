import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { SaveImageInvocation } from 'services/api/types';
import {
  CANVAS_OUTPUT,
  LATENTS_TO_IMAGE,
  NSFW_CHECKER,
  SAVE_IMAGE,
  WATERMARKER,
} from './constants';

/**
 * Set the `use_cache` field on the linear/canvas graph's final image output node to False.
 */
export const addSaveImageNode = (
  state: RootState,
  graph: NonNullableGraph
): void => {
  const activeTabName = activeTabNameSelector(state);
  const is_intermediate =
    activeTabName === 'unifiedCanvas' ? !state.canvas.shouldAutoSave : false;
  const { autoAddBoardId } = state.gallery;

  const saveImageNode: SaveImageInvocation = {
    id: SAVE_IMAGE,
    type: 'save_image',
    is_intermediate,
    use_cache: false,
    board: autoAddBoardId === 'none' ? undefined : { board_id: autoAddBoardId },
  };

  graph.nodes[SAVE_IMAGE] = saveImageNode;

  const destination = {
    node_id: SAVE_IMAGE,
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
