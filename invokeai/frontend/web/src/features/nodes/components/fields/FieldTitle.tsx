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
import { MouseEvent, memo, useCallback, useMemo, useState } from 'react';
import { NodeProps } from 'reactflow';

interface Props {
  nodeProps: NodeProps<InvocationNodeData>;
  nodeTemplate: InvocationTemplate;
  field: InputFieldValue;
  fieldTemplate: InputFieldTemplate;
}

const FieldTitle = (props: Props) => {
  const { nodeProps, nodeTemplate, field, fieldTemplate } = props;
  const { label } = field;
  const { title, input } = fieldTemplate;
  const { id: nodeId } = nodeProps.data;
  const dispatch = useAppDispatch();
  const [localTitle, setLocalTitle] = useState(label || title);

  const draggableData: NodeFieldDraggableData | undefined = useMemo(
    () =>
      input !== 'connection'
        ? {
            id: `${nodeId}-${field.name}`,
            payloadType: 'NODE_FIELD',
            payload: { nodeId, field, fieldTemplate },
          }
        : undefined,
    [field, fieldTemplate, input, nodeId]
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

  return (
    <Flex
      className="nopan"
      sx={{
        position: 'relative',
        overflow: 'hidden',
        w: 'full',
        h: 'full',
        alignItems: 'flex-start',
        justifyContent: 'flex-start',
        cursor: input !== 'connection' ? 'grab' : 'text',
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
            textAlign: 'left',
          }}
          noOfLines={1}
        />
        <EditableInput
          fontSize="sm"
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

  return (
    <IAIDraggable
      disabled={!props.draggableData}
      data={props.draggableData}
      onDoubleClick={handleDoubleClick}
      cursor={props.draggableData ? 'grab' : 'text'}
    />
  );
}
