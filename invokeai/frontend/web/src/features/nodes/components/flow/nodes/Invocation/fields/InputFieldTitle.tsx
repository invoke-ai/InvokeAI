import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Editable, EditableInput, EditablePreview, Flex, Tooltip, useEditableControls } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { InputFieldTooltip } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldTooltip';
import { useInputFieldLabel } from 'features/nodes/hooks/useInputFieldLabel';
import { useInputFieldTemplateTitle } from 'features/nodes/hooks/useInputFieldTemplateTitle';
import { fieldLabelChanged } from 'features/nodes/store/nodesSlice';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import type { MouseEvent } from 'react';
import { memo, useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

const editablePreviewStyles: SystemStyleObject = {
  p: 0,
  fontWeight: 'semibold',
  textAlign: 'left',
  color: 'base.300',
  _hover: {
    fontWeight: 'semibold !important',
  },
  '&[data-is-invalid="true"]': {
    color: 'error.300',
  },
  '&[data-is-disabled="true"]': {
    opacity: 0.5,
  },
};

const editableInputStyles: SystemStyleObject = {
  p: 0,
  w: 'full',
  fontWeight: 'semibold',
  color: 'base.100',
  _focusVisible: {
    p: 0,
    textAlign: 'left',
    boxShadow: 'none',
  },
};

interface Props {
  nodeId: string;
  fieldName: string;
  isInvalid?: boolean;
  isDisabled?: boolean;
}

export const InputFieldTitle = memo((props: Props) => {
  const { nodeId, fieldName, isInvalid, isDisabled } = props;
  const label = useInputFieldLabel(nodeId, fieldName);
  const fieldTemplateTitle = useInputFieldTemplateTitle(nodeId, fieldName);
  const { t } = useTranslation();

  const dispatch = useAppDispatch();
  const [localTitle, setLocalTitle] = useState(label || fieldTemplateTitle || t('nodes.unknownField'));

  const handleSubmit = useCallback(
    (newTitleRaw: string) => {
      const newTitle = newTitleRaw.trim();
      const finalTitle = newTitle || fieldTemplateTitle || t('nodes.unknownField');
      setLocalTitle(finalTitle);
      dispatch(fieldLabelChanged({ nodeId, fieldName, label: finalTitle }));
    },
    [fieldTemplateTitle, dispatch, nodeId, fieldName, t]
  );

  const handleChange = useCallback((newTitle: string) => {
    setLocalTitle(newTitle);
  }, []);

  useEffect(() => {
    // Another component may change the title; sync local title with global state
    setLocalTitle(label || fieldTemplateTitle || t('nodes.unknownField'));
  }, [label, fieldTemplateTitle, t]);

  return (
    <Editable
      value={localTitle}
      onChange={handleChange}
      onSubmit={handleSubmit}
      as={Flex}
      position="relative"
      overflow="hidden"
      alignItems="center"
      justifyContent="flex-start"
      gap={1}
      w="full"
    >
      <Tooltip
        label={<InputFieldTooltip nodeId={nodeId} fieldName={fieldName} />}
        openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
      >
        <EditablePreview
          sx={editablePreviewStyles}
          noOfLines={1}
          data-is-invalid={isInvalid}
          data-is-disabled={isDisabled}
        />
      </Tooltip>
      <EditableInput className="nodrag" sx={editableInputStyles} />
      <EditableControls />
    </Editable>
  );
});

InputFieldTitle.displayName = 'InputFieldTitle';

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
      w="min-content"
      h="full"
      top={0}
      insetInlineStart={0}
      cursor="text"
    />
  );
});

EditableControls.displayName = 'EditableControls';
