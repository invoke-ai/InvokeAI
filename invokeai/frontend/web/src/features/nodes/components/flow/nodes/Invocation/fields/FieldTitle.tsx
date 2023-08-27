import {
  Editable,
  EditableInput,
  EditablePreview,
  Flex,
  forwardRef,
  useEditableControls,
} from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { useFieldLabel } from 'features/nodes/hooks/useFieldLabel';
import { useFieldTemplateTitle } from 'features/nodes/hooks/useFieldTemplateTitle';
import { fieldLabelChanged } from 'features/nodes/store/nodesSlice';
import { MouseEvent, memo, useCallback, useEffect, useState } from 'react';

interface Props {
  nodeId: string;
  fieldName: string;
  kind: 'input' | 'output';
  isMissingInput?: boolean;
}

const FieldTitle = forwardRef((props: Props, ref) => {
  const { nodeId, fieldName, kind, isMissingInput = false } = props;
  const label = useFieldLabel(nodeId, fieldName);
  const fieldTemplateTitle = useFieldTemplateTitle(nodeId, fieldName, kind);

  const dispatch = useAppDispatch();
  const [localTitle, setLocalTitle] = useState(
    label || fieldTemplateTitle || 'Unknown Field'
  );

  const handleSubmit = useCallback(
    async (newTitle: string) => {
      if (newTitle && (newTitle === label || newTitle === fieldTemplateTitle)) {
        return;
      }
      setLocalTitle(newTitle || fieldTemplateTitle || 'Unknown Field');
      dispatch(fieldLabelChanged({ nodeId, fieldName, label: newTitle }));
    },
    [label, fieldTemplateTitle, dispatch, nodeId, fieldName]
  );

  const handleChange = useCallback((newTitle: string) => {
    setLocalTitle(newTitle);
  }, []);

  useEffect(() => {
    // Another component may change the title; sync local title with global state
    setLocalTitle(label || fieldTemplateTitle || 'Unknown Field');
  }, [label, fieldTemplateTitle]);

  return (
    <Flex
      ref={ref}
      sx={{
        position: 'relative',
        overflow: 'hidden',
        alignItems: 'center',
        justifyContent: 'flex-start',
        gap: 1,
        h: 'full',
        w: 'full',
      }}
    >
      <Editable
        value={localTitle}
        onChange={handleChange}
        onSubmit={handleSubmit}
        as={Flex}
        sx={{
          position: 'relative',
          alignItems: 'center',
          h: 'full',
          w: 'full',
        }}
      >
        <EditablePreview
          sx={{
            p: 0,
            fontWeight: isMissingInput ? 600 : 400,
            textAlign: 'left',
            _hover: {
              fontWeight: '600 !important',
            },
          }}
          noOfLines={1}
        />
        <EditableInput
          className="nodrag"
          sx={{
            p: 0,
            fontWeight: 600,
            color: 'base.900',
            _dark: {
              color: 'base.100',
            },
            _focusVisible: {
              p: 0,
              textAlign: 'left',
              boxShadow: 'none',
            },
          }}
        />
        <EditableControls />
      </Editable>
    </Flex>
  );
});

export default memo(FieldTitle);

const EditableControls = memo(() => {
  const { isEditing, getEditButtonProps } = useEditableControls();
  const handleClick = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      const { onClick } = getEditButtonProps();
      if (!onClick) {
        return;
      }
      onClick(e);
      e.preventDefault();
    },
    [getEditButtonProps]
  );

  if (isEditing) {
    return null;
  }

  return (
    <Flex
      onClick={handleClick}
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
