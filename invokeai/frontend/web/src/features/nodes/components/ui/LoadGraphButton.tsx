import { FileButton } from '@mantine/core';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  loadFileEdges,
  loadFileNodes,
  workflowLoaded,
} from 'features/nodes/store/nodesSlice';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import i18n from 'i18n';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { FaUpload } from 'react-icons/fa';
import { useReactFlow } from 'reactflow';

interface JsonFile {
  [key: string]: unknown;
}

function sanityCheckInvokeAIGraph(jsonFile: JsonFile): {
  isValid: boolean;
  message: string;
} {
  // Check if primary keys exist
  const keys = ['nodes', 'edges', 'viewport'];
  for (const key of keys) {
    if (!(key in jsonFile)) {
      return {
        isValid: false,
        message: i18n.t('toast.nodesNotValidGraph'),
      };
    }
  }

  // Check if nodes and edges are arrays
  if (!Array.isArray(jsonFile.nodes) || !Array.isArray(jsonFile.edges)) {
    return {
      isValid: false,
      message: i18n.t('toast.nodesNotValidGraph'),
    };
  }

  // Check if data is present in nodes
  const nodeKeys = ['data', 'type'];
  const nodeTypes = ['invocation', 'current_image'];
  if (jsonFile.nodes.length > 0) {
    for (const node of jsonFile.nodes) {
      for (const nodeKey of nodeKeys) {
        if (!(nodeKey in node)) {
          return {
            isValid: false,
            message: i18n.t('toast.nodesNotValidGraph'),
          };
        }
        if (nodeKey === 'type' && !nodeTypes.includes(node[nodeKey])) {
          return {
            isValid: false,
            message: i18n.t('toast.nodesUnrecognizedTypes'),
          };
        }
      }
    }
  }

  // Check Edge Object
  const edgeKeys = ['source', 'sourceHandle', 'target', 'targetHandle'];
  if (jsonFile.edges.length > 0) {
    for (const edge of jsonFile.edges) {
      for (const edgeKey of edgeKeys) {
        if (!(edgeKey in edge)) {
          return {
            isValid: false,
            message: i18n.t('toast.nodesBrokenConnections'),
          };
        }
      }
    }
  }

  return {
    isValid: true,
    message: i18n.t('toast.nodesLoaded'),
  };
}

const LoadGraphButton = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { fitView } = useReactFlow();

  const uploadedFileRef = useRef<() => void>(null);

  const restoreJSONToEditor = useCallback(
    (v: File | null) => {
      if (!v) return;
      const reader = new FileReader();
      reader.onload = async () => {
        const json = reader.result;

        dispatch(workflowLoaded(JSON.parse(json as string)));
        // try {
        //   const retrievedNodeTree = await JSON.parse(String(json));
        //   const { isValid, message } =
        //     sanityCheckInvokeAIGraph(retrievedNodeTree);

        //   if (isValid) {
        //     dispatch(loadFileNodes(retrievedNodeTree.nodes));
        //     dispatch(loadFileEdges(retrievedNodeTree.edges));
        //     fitView();

        //     dispatch(
        //       addToast(makeToast({ title: message, status: 'success' }))
        //     );
        //   } else {
        //     dispatch(
        //       addToast(
        //         makeToast({
        //           title: message,
        //           status: 'error',
        //         })
        //       )
        //     );
        //   }
        //   // Cleanup
        //   reader.abort();
        // } catch (error) {
        //   if (error) {
        //     dispatch(
        //       addToast(
        //         makeToast({
        //           title: t('toast.nodesNotValidJSON'),
        //           status: 'error',
        //         })
        //       )
        //     );
        //   }
        // }
      };

      reader.readAsText(v);

      // Cleanup
      uploadedFileRef.current?.();
    },
    [dispatch]
  );
  return (
    <FileButton
      resetRef={uploadedFileRef}
      accept="application/json"
      onChange={restoreJSONToEditor}
    >
      {(props) => (
        <IAIIconButton
          icon={<FaUpload />}
          tooltip={t('nodes.loadGraph')}
          aria-label={t('nodes.loadGraph')}
          {...props}
        />
      )}
    </FileButton>
  );
};

export default memo(LoadGraphButton);
