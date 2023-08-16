import {
  Box,
  Editable,
  EditableInput,
  EditablePreview,
  Flex,
  useEditableControls,
} from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import {
  useNodeLabel,
  useNodeTemplateTitle,
} from 'features/nodes/hooks/useNodeData';
import { nodeLabelChanged } from 'features/nodes/store/nodesSlice';
import { DRAG_HANDLE_CLASSNAME } from 'features/nodes/types/constants';
import { MouseEvent, memo, useCallback, useEffect, useState } from 'react';

type Props = {
  nodeId: string;
  title?: string;
};

const NodeTitle = ({ nodeId, title }: Props) => {
  const dispatch = useAppDispatch();
  const label = useNodeLabel(nodeId);
  const templateTitle = useNodeTemplateTitle(nodeId);

  const [localTitle, setLocalTitle] = useState('');
  const handleSubmit = useCallback(
    async (newTitle: string) => {
      dispatch(nodeLabelChanged({ nodeId, label: newTitle }));
      setLocalTitle(newTitle || title || 'Problem Setting Title');
    },
    [nodeId, dispatch, title]
  );

  const handleChange = useCallback((newTitle: string) => {
    setLocalTitle(newTitle);
  }, []);

  useEffect(() => {
    // Another component may change the title; sync local title with global state
    setLocalTitle(label || title || templateTitle || 'Problem Setting Title');
  }, [label, templateTitle, title]);

  return (
    <Flex
      className="nopan"
      sx={{
        overflow: 'hidden',
        w: 'full',
        h: 'full',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: 'text',
      }}
    >
      <Editable
        as={Flex}
        value={localTitle}
        onChange={handleChange}
        onSubmit={handleSubmit}
        sx={{
          alignItems: 'center',
          position: 'relative',
          w: 'full',
          h: 'full',
        }}
      >
        <EditablePreview
          fontSize="sm"
          sx={{
            p: 0,
            w: 'full',
          }}
          noOfLines={1}
        />
        <EditableInput
          fontSize="sm"
          sx={{
            p: 0,
            _focusVisible: {
              p: 0,
              boxShadow: 'none',
            },
          }}
        />
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
      sx={{
        position: 'absolute',
        w: 'full',
        h: 'full',
        top: 0,
        cursor: 'grab',
      }}
    />
  );
}
