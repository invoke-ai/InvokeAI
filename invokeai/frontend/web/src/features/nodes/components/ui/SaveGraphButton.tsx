import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { map, omit } from 'lodash-es';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaSave } from 'react-icons/fa';

const SaveGraphButton = () => {
  const { t } = useTranslation();
  const editorInstance = useAppSelector(
    (state: RootState) => state.nodes.editorInstance
  );

  const nodes = useAppSelector((state: RootState) => state.nodes.nodes);

  const saveEditorToJSON = useCallback(() => {
    if (editorInstance) {
      const editorState = editorInstance.toObject();

      editorState.edges = map(editorState.edges, (edge) => {
        return omit(edge, ['style']);
      });

      const nodeSetupJSON = new Blob([JSON.stringify(editorState)]);
      const nodeDownloadElement = document.createElement('a');
      nodeDownloadElement.href = URL.createObjectURL(nodeSetupJSON);
      nodeDownloadElement.download = 'MyNodes.json';
      document.body.appendChild(nodeDownloadElement);
      nodeDownloadElement.click();
      // Cleanup
      nodeDownloadElement.remove();
    }
  }, [editorInstance]);

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
