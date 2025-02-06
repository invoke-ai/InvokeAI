import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Input, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEditable } from 'common/hooks/useEditable';
import { InputFieldTooltipContent } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldTooltipContent';
import { useInputFieldLabel } from 'features/nodes/hooks/useInputFieldLabel';
import { useInputFieldTemplateTitle } from 'features/nodes/hooks/useInputFieldTemplateTitle';
import { fieldLabelChanged } from 'features/nodes/store/nodesSlice';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import { memo, useCallback, useMemo, useRef } from 'react';
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
  const defaultTitle = useMemo(() => fieldTemplateTitle || t('nodes.unknownField'), [fieldTemplateTitle, t]);
  const onChange = useCallback(
    (label: string) => {
      dispatch(fieldLabelChanged({ nodeId, fieldName, label }));
    },
    [dispatch, nodeId, fieldName]
  );
  const editable = useEditable({
    value: label || defaultTitle,
    defaultValue: defaultTitle,
    onChange,
    inputRef,
  });

  if (!editable.isEditing) {
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
          onDoubleClick={editable.startEditing}
        >
          {editable.value}
        </Text>
      </Tooltip>
    );
  }

  return <Input ref={inputRef} variant="outline" {...editable.inputProps} />;
});

InputFieldTitle.displayName = 'InputFieldTitle';
