import {
  Editable,
  EditableInput,
  EditablePreview,
  Flex,
  useEditableControls,
} from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIDraggable from 'common/components/IAIDraggable';
import { NodeFieldDraggableData } from 'features/dnd/types';
import { fieldLabelChanged } from 'features/nodes/store/nodesSlice';
import {
  InputFieldTemplate,
  InputFieldValue,
  InvocationNodeData,
  InvocationTemplate,
} from 'features/nodes/types/types';
import {
  MouseEvent,
  memo,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from 'react';

interface Props {
  nodeData: InvocationNodeData;
  nodeTemplate: InvocationTemplate;
  field: InputFieldValue;
  fieldTemplate: InputFieldTemplate;
  isDraggable?: boolean;
}

const FieldTitle = (props: Props) => {
  const { nodeData, field, fieldTemplate, isDraggable = false } = props;
  const { label } = field;
  const { title, input } = fieldTemplate;
  const { id: nodeId } = nodeData;
  const dispatch = useAppDispatch();
  const [localTitle, setLocalTitle] = useState(label || title);

  const draggableData: NodeFieldDraggableData | undefined = useMemo(
    () =>
      input !== 'connection' && isDraggable
        ? {
            id: `${nodeId}-${field.name}`,
            payloadType: 'NODE_FIELD',
            payload: { nodeId, field, fieldTemplate },
          }
        : undefined,
    [field, fieldTemplate, input, isDraggable, nodeId]
  );

  const handleSubmit = useCallback(
    async (newTitle: string) => {
      dispatch(
        fieldLabelChanged({ nodeId, fieldName: field.name, label: newTitle })
      );
      setLocalTitle(newTitle || title);
    },
    [dispatch, nodeId, field.name, title]
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
        position: 'relative',
        overflow: 'hidden',
        h: 'full',
        alignItems: 'flex-start',
        justifyContent: 'flex-start',
        gap: 1,
      }}
    >
      <Editable
        value={localTitle}
        onChange={handleChange}
        onSubmit={handleSubmit}
        sx={{
          position: 'relative',
        }}
      >
        <EditablePreview
          sx={{
            p: 0,
            textAlign: 'left',
          }}
          noOfLines={1}
        />
        <EditableInput
          sx={{
            p: 0,
            _focusVisible: {
              p: 0,
              textAlign: 'left',
              boxShadow: 'none',
            },
          }}
        />
        <EditableControls draggableData={draggableData} />
      </Editable>
    </Flex>
  );
};

export default memo(FieldTitle);

type EditableControlsProps = {
  draggableData?: NodeFieldDraggableData;
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
    },
    [getEditButtonProps]
  );

  if (isEditing) {
    return null;
  }

  if (props.draggableData) {
    return (
      <IAIDraggable
        data={props.draggableData}
        onDoubleClick={handleDoubleClick}
        cursor={props.draggableData ? 'grab' : 'text'}
      />
    );
  }

  return (
    <Flex
      onDoubleClick={handleDoubleClick}
      position="absolute"
      w="full"
      h="full"
      top={0}
      insetInlineStart={0}
      cursor="text"
    />
  );
}
