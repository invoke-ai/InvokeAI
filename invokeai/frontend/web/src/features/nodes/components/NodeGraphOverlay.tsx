import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import ImageMetadataJSON from 'features/gallery/components/ImageMetadataViewer/ImageMetadataJSON';
import { omit } from 'lodash-es';
import { useMemo } from 'react';
import { useDebounce } from 'use-debounce';
import { buildNodesGraph } from '../util/graphBuilders/buildNodesGraph';

const useNodesGraph = () => {
  const nodes = useAppSelector((state: RootState) => state.nodes);
  const [debouncedNodes] = useDebounce(nodes, 300);
  const graph = useMemo(
    () => omit(buildNodesGraph(debouncedNodes), 'id'),
    [debouncedNodes]
  );

  return graph;
};

const NodeGraph = () => {
  const graph = useNodesGraph();

  return <ImageMetadataJSON jsonObject={graph} label="Graph" />;
};

export default NodeGraph;
