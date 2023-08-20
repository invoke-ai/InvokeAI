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
import {
  useFieldData,
  useFieldTemplate,
} from 'features/nodes/hooks/useNodeData';
import { fieldLabelChanged } from 'features/nodes/store/nodesSlice';
import {
  MouseEvent,
  memo,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from 'react';

interface Props {
  nodeId: string;
  fieldName: string;
  isDraggable?: boolean;
  kind: 'input' | 'output';
}

const FieldTitle = (props: Props) => {
  const { nodeId, fieldName, isDraggable = false, kind } = props;
  const fieldTemplate = useFieldTemplate(nodeId, fieldName, kind);
  const field = useFieldData(nodeId, fieldName);

  const dispatch = useAppDispatch();
  const [localTitle, setLocalTitle] = useState(
    field?.label || fieldTemplate?.title || 'Unknown Field'
  );

  const draggableData: NodeFieldDraggableData | undefined = useMemo(
    () =>
      field &&
      fieldTemplate?.fieldKind === 'input' &&
      fieldTemplate?.input !== 'connection' &&
      isDraggable
        ? {
            id: `${nodeId}-${fieldName}`,
            payloadType: 'NODE_FIELD',
            payload: { nodeId, field, fieldTemplate },
          }
        : undefined,
    [field, fieldName, fieldTemplate, isDraggable, nodeId]
  );

  const handleSubmit = useCallback(
    async (newTitle: string) => {
      dispatch(fieldLabelChanged({ nodeId, fieldName, label: newTitle }));
      setLocalTitle(newTitle || fieldTemplate?.title || 'Unknown Field');
    },
    [dispatch, nodeId, fieldName, fieldTemplate?.title]
  );

  const handleChange = useCallback((newTitle: string) => {
    setLocalTitle(newTitle);
  }, []);

  useEffect(() => {
    // Another component may change the title; sync local title with global state
    setLocalTitle(field?.label || fieldTemplate?.title || 'Unknown Field');
  }, [field?.label, fieldTemplate?.title]);

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

const EditableControls = memo((props: EditableControlsProps) => {
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
});

EditableControls.displayName = 'EditableControls';
