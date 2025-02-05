import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Input, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { InputFieldTooltipContent } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldTooltipContent';
import { useInputFieldLabel } from 'features/nodes/hooks/useInputFieldLabel';
import { useInputFieldTemplateTitle } from 'features/nodes/hooks/useInputFieldTemplateTitle';
import { fieldLabelChanged } from 'features/nodes/store/nodesSlice';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

const labelSx: SystemStyleObject = {
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

interface Props {
  nodeId: string;
  fieldName: string;
  isInvalid?: boolean;
  isDisabled?: boolean;
}

export const InputFieldTitle = memo((props: Props) => {
  const { nodeId, fieldName, isInvalid, isDisabled } = props;
  const inputRef = useRef<HTMLInputElement>(null);
  const label = useInputFieldLabel(nodeId, fieldName);
  const fieldTemplateTitle = useInputFieldTemplateTitle(nodeId, fieldName);
  const { t } = useTranslation();

  const dispatch = useAppDispatch();
  const [isEditing, setIsEditing] = useState(false);
  const [localTitle, setLocalTitle] = useState(label || fieldTemplateTitle || t('nodes.unknownField'));

  const onBlur = useCallback(() => {
    const trimmedTitle = localTitle.trim();
    const finalTitle = trimmedTitle || fieldTemplateTitle || t('nodes.unknownField');
    if (trimmedTitle !== localTitle) {
      setLocalTitle(finalTitle);
      dispatch(fieldLabelChanged({ nodeId, fieldName, label: finalTitle }));
    }
    setIsEditing(false);
  }, [localTitle, fieldTemplateTitle, t, dispatch, nodeId, fieldName]);

  const handleChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setLocalTitle(e.target.value);
  }, []);

  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        onBlur();
      } else if (e.key === 'Escape') {
        setLocalTitle(label || fieldTemplateTitle || t('nodes.unknownField'));
        setIsEditing(false);
      }
    },
    [fieldTemplateTitle, label, onBlur, t]
  );

  const onEdit = useCallback(() => {
    setIsEditing(true);
  }, []);

  useEffect(() => {
    // Another component may change the title; sync local title with global state
    setLocalTitle(label || fieldTemplateTitle || t('nodes.unknownField'));
  }, [label, fieldTemplateTitle, t]);

  useEffect(() => {
    if (isEditing) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [isEditing]);

  if (!isEditing) {
    return (
      <Tooltip
        label={<InputFieldTooltipContent nodeId={nodeId} fieldName={fieldName} />}
        openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
        placement="top"
      >
        <Text
          sx={labelSx}
          noOfLines={1}
          data-is-invalid={isInvalid}
          data-is-disabled={isDisabled}
          onDoubleClick={onEdit}
        >
          {localTitle}
        </Text>
      </Tooltip>
    );
  }

  return (
    <Input
      ref={inputRef}
      value={localTitle}
      onChange={handleChange}
      onBlur={onBlur}
      onKeyDown={onKeyDown}
      variant="unstyled"
    />
  );
});

InputFieldTitle.displayName = 'InputFieldTitle';
