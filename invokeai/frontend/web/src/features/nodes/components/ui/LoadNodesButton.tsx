import { FileButton } from '@mantine/core';
import { makeToast } from 'features/system/util/makeToast';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { loadFileEdges, loadFileNodes } from 'features/nodes/store/nodesSlice';
import { addToast } from 'features/system/store/systemSlice';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { FaUpload } from 'react-icons/fa';
import { useReactFlow } from 'reactflow';

const LoadNodesButton = () => {
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
        const retrievedNodeTree = await JSON.parse(String(json));

        if (!retrievedNodeTree) {
          dispatch(
            addToast(
              makeToast({
                title: t('toast.nodesLoadedFailed'),
                status: 'error',
              })
            )
          );
        }

        if (retrievedNodeTree) {
          dispatch(loadFileNodes(retrievedNodeTree.nodes));
          dispatch(loadFileEdges(retrievedNodeTree.edges));
          fitView();

          dispatch(
            addToast(
              makeToast({ title: t('toast.nodesLoaded'), status: 'success' })
            )
          );
        }

        // Cleanup
        reader.abort();
      };

      reader.readAsText(v);

      // Cleanup
      uploadedFileRef.current?.();
    },
    [fitView, dispatch, t]
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
          tooltip={t('nodes.loadNodes')}
          aria-label={t('nodes.loadNodes')}
          {...props}
        />
      )}
    </FileButton>
  );
};

export default memo(LoadNodesButton);
