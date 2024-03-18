import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Editable, EditableInput, EditablePreview, Flex, useEditableControls } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useNodeLabel } from 'features/nodes/hooks/useNodeLabel';
import { useNodeTemplateTitle } from 'features/nodes/hooks/useNodeTemplateTitle';
import { nodeLabelChanged } from 'features/nodes/store/nodesSlice';
import { DRAG_HANDLE_CLASSNAME } from 'features/nodes/types/constants';
import type { MouseEvent } from 'react';
import { memo, useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  nodeId: string;
  title?: string;
};

const NodeTitle = ({ nodeId, title }: Props) => {
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
    <Flex overflow="hidden" w="full" h="full" alignItems="center" justifyContent="center" cursor="text">
      <Editable
        as={Flex}
        value={localTitle}
        onChange={handleChange}
        onSubmit={handleSubmit}
        alignItems="center"
        position="relative"
        w="full"
        h="full"
      >
        <EditablePreview fontSize="sm" p={0} w="full" noOfLines={1} />
        <EditableInput className="nodrag" fontSize="sm" sx={editableInputStyles} />
        <EditableControls />
      </Editable>
    </Flex>
  );
};

export default memo(NodeTitle);

function EditableControls() {
  const { isEditing, getEditButtonProps } = useEditableControls();
  const handleDoubleClick = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      const { onClick } = getEditButtonProps();
      if (!onClick) {
        return;
      }
      onClick(e);
    },
    [getEditButtonProps]
  );

  if (isEditing) {
    return null;
  }

  return (
    <Box
      className={DRAG_HANDLE_CLASSNAME}
      onDoubleClick={handleDoubleClick}
      position="absolute"
      w="full"
      h="full"
      top={0}
      cursor="grab"
    />
  );
}

const editableInputStyles: SystemStyleObject = {
  p: 0,
  fontWeight: 'bold',
  _focusVisible: {
    p: 0,
  },
};
