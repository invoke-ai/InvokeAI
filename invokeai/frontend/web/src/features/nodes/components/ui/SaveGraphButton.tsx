import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { buildWorkflow } from 'features/nodes/util/buildWorkflow';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaSave } from 'react-icons/fa';

const SaveGraphButton = () => {
  const { t } = useTranslation();
  // const editorInstance = useAppSelector(
  //   (state: RootState) => state.nodes.editorInstance
  // );

  const nodesState = useAppSelector((state) => state.nodes);
  const nodes = useAppSelector((state: RootState) => state.nodes.nodes);

  const saveEditorToJSON = useCallback(() => {
    const workflow = buildWorkflow(nodesState);
    const workflowJSON = new Blob([JSON.stringify(workflow)]);
    const workflowDownloadElement = document.createElement('a');
    workflowDownloadElement.href = URL.createObjectURL(workflowJSON);
    workflowDownloadElement.download = 'MyNodes.json';
    document.body.appendChild(workflowDownloadElement);
    workflowDownloadElement.click();
    // Cleanup
    workflowDownloadElement.remove();
  }, [nodesState]);
  // const saveEditorToJSON = useCallback(() => {
  //   if (editorInstance) {
  //     const editorState = editorInstance.toObject();

  //     editorState.edges = map(editorState.edges, (edge) => {
  //       return omit(edge, ['style']);
  //     });

  //     const nodeSetupJSON = new Blob([JSON.stringify(editorState)]);
  //     const nodeDownloadElement = document.createElement('a');
  //     nodeDownloadElement.href = URL.createObjectURL(nodeSetupJSON);
  //     nodeDownloadElement.download = 'MyNodes.json';
  //     document.body.appendChild(nodeDownloadElement);
  //     nodeDownloadElement.click();
  //     // Cleanup
  //     nodeDownloadElement.remove();
  //   }
  // }, [editorInstance]);

  return (
    <IAIIconButton
      icon={<FaSave />}
      fontSize={18}
      tooltip={t('nodes.saveGraph')}
      aria-label={t('nodes.saveGraph')}
      onClick={saveEditorToJSON}
      isDisabled={nodes.length === 0}
    />
  );
};

export default memo(SaveGraphButton);
