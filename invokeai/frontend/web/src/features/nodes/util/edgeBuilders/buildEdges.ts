import {
  Edge,
  ImageToImageInvocation,
  InpaintInvocation,
  IterateInvocation,
  RandomRangeInvocation,
  RangeInvocation,
  TextToImageInvocation,
} from 'services/api';

export const buildEdges = (
  baseNode: TextToImageInvocation | ImageToImageInvocation | InpaintInvocation,
  rangeNode: RangeInvocation | RandomRangeInvocation,
  iterateNode: IterateInvocation
): Edge[] => {
  const edges: Edge[] = [
    {
      source: {
        node_id: rangeNode.id,
        field: 'collection',
      },
      destination: {
        node_id: iterateNode.id,
        field: 'collection',
      },
    },
    {
      source: {
        node_id: iterateNode.id,
        field: 'item',
      },
      destination: {
        node_id: baseNode.id,
        field: 'seed',
      },
    },
  ];

  return edges;
};
