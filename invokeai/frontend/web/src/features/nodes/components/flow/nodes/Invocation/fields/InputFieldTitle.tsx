import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Input, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEditable } from 'common/hooks/useEditable';
import { InputFieldTooltipContent } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldTooltipContent';
import {
  useConnectionErrorTKey,
  useIsConnectionInProgress,
  useIsConnectionStartField,
} from 'features/nodes/hooks/useFieldConnectionState';
import { useInputFieldIsConnected } from 'features/nodes/hooks/useInputFieldIsConnected';
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
}

export const InputFieldTitle = memo((props: Props) => {
  const { nodeId, fieldName, isInvalid } = props;
  const inputRef = useRef<HTMLInputElement>(null);
  const label = useInputFieldLabel(nodeId, fieldName);
  const fieldTemplateTitle = useInputFieldTemplateTitle(nodeId, fieldName);
  const { t } = useTranslation();
  const isConnected = useInputFieldIsConnected(nodeId, fieldName);
  const isConnectionStartField = useIsConnectionStartField(nodeId, fieldName, 'target');
  const isConnectionInProgress = useIsConnectionInProgress();
  const connectionError = useConnectionErrorTKey(nodeId, fieldName, 'target');

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
          data-is-disabled={
            (isConnectionInProgress && connectionError !== null && !isConnectionStartField) || isConnected
          }
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
