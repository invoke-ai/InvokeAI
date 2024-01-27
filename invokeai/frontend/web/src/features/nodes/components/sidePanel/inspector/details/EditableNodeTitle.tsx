import { Editable, EditableInput, EditablePreview, Flex } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useNodeLabel } from 'features/nodes/hooks/useNodeLabel';
import { useNodeTemplateTitle } from 'features/nodes/hooks/useNodeTemplateTitle';
import { nodeLabelChanged } from 'features/nodes/store/nodesSlice';
import { memo, useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  nodeId: string;
  title?: string;
};

const EditableNodeTitle = ({ nodeId, title }: Props) => {
  const dispatch = useAppDispatch();
  const label = useNodeLabel(nodeId);
  const templateTitle = useNodeTemplateTitle(nodeId);
  const { t } = useTranslation();

  const [localTitle, setLocalTitle] = useState('');
  const handleSubmit = useCallback(
    async (newTitle: string) => {
      dispatch(nodeLabelChanged({ nodeId, label: newTitle }));
      setLocalTitle(label || title || templateTitle || t('nodes.problemSettingTitle'));
    },
    [dispatch, nodeId, title, templateTitle, label, t]
  );

  const handleChange = useCallback((newTitle: string) => {
    setLocalTitle(newTitle);
  }, []);

  useEffect(() => {
    // Another component may change the title; sync local title with global state
    setLocalTitle(label || title || templateTitle || t('nodes.problemSettingTitle'));
  }, [label, templateTitle, title, t]);

  return (
    <Flex w="full" alignItems="center" justifyContent="center">
      <Editable
        as={Flex}
        value={localTitle}
        onChange={handleChange}
        onSubmit={handleSubmit}
        w="full"
        fontWeight="semibold"
      >
        <EditablePreview noOfLines={1} />
        <EditableInput className="nodrag" />
      </Editable>
    </Flex>
  );
};

export default memo(EditableNodeTitle);
