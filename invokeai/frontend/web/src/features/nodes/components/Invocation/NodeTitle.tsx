import {
  Box,
  Editable,
  EditableInput,
  EditablePreview,
  Flex,
  useEditableControls,
} from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { nodeLabelChanged } from 'features/nodes/store/nodesSlice';
import { DRAG_HANDLE_CLASSNAME } from 'features/nodes/types/constants';
import { NodeData } from 'features/nodes/types/types';
import { MouseEvent, memo, useCallback, useEffect, useState } from 'react';

type Props = {
  nodeData: NodeData;
  title: string;
};

const NodeTitle = (props: Props) => {
  const { title } = props;
  const { id: nodeId, label } = props.nodeData;
  const dispatch = useAppDispatch();
  const [localTitle, setLocalTitle] = useState(label || title);

  const handleSubmit = useCallback(
    async (newTitle: string) => {
      dispatch(nodeLabelChanged({ nodeId, label: newTitle }));
      setLocalTitle(newTitle || title);
    },
    [nodeId, dispatch, title]
  );

  const handleChange = useCallback((newTitle: string) => {
    setLocalTitle(newTitle);
  }, []);

  useEffect(() => {
    // Another component may change the title; sync local title with global state
    setLocalTitle(label || title);
  }, [label, title]);

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
