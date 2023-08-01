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
import { MouseEvent, memo, useCallback, useState } from 'react';
import { NodeProps } from 'reactflow';

interface Props {
  nodeProps: NodeProps<NodeData>;
  title: string;
}

const NodeTitle = (props: Props) => {
  const { title } = props;
  const { data } = props.nodeProps;
  const dispatch = useAppDispatch();
  const [isEditing, setIsEditing] = useState(false);
  const [localTitle, setLocalTitle] = useState(data.label || title);

  const handleSubmit = useCallback(
    async (newTitle: string) => {
      setIsEditing(false);
      dispatch(nodeLabelChanged({ nodeId: data.id, label: newTitle }));
      setLocalTitle(newTitle || title);
    },
    [data.id, dispatch, title]
  );

  const handleChange = useCallback((newTitle: string) => {
    setLocalTitle(newTitle);
  }, []);

  return (
    <Flex
      className={isEditing ? 'nopan' : DRAG_HANDLE_CLASSNAME}
      sx={{
        overflow: 'hidden',
        w: 'full',
        h: 'full',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: isEditing ? 'text' : undefined,
      }}
    >
      <Editable
        value={localTitle}
        onChange={handleChange}
        onSubmit={handleSubmit}
        sx={{
          position: 'relative',
          w: 'full',
        }}
      >
        <EditablePreview
          fontSize="sm"
          sx={{
            p: 0,
            textAlign: 'center',
            fontWeight: 600,
            color: 'base.700',
            _dark: { color: 'base.200' },
          }}
          noOfLines={1}
        />
        <EditableInput
          fontSize="sm"
          sx={{
            p: 0,
            fontWeight: 600,
            _focusVisible: {
              p: 0,
              textAlign: 'center',
              boxShadow: 'none',
            },
          }}
        />
        <EditableControls setIsEditing={setIsEditing} />
      </Editable>
    </Flex>
  );
};

export default memo(NodeTitle);

type EditableControlsProps = {
  setIsEditing: (isEditing: boolean) => void;
};

function EditableControls(props: EditableControlsProps) {
  const { isEditing, getEditButtonProps } = useEditableControls();
  const handleDoubleClick = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      const { onClick } = getEditButtonProps();
      if (!onClick) {
        return;
      }
      onClick(e);
      props.setIsEditing(true);
    },
    [getEditButtonProps, props]
  );

  return isEditing ? null : (
    <Box
      onDoubleClick={handleDoubleClick}
      sx={{
        position: 'absolute',
        w: 'full',
        h: 'full',
        top: 0,
      }}
    />
  );
}
