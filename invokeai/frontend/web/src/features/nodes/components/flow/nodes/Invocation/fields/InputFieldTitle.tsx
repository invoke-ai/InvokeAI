import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Icon, Input, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEditable } from 'common/hooks/useEditable';
import { InputFieldTooltipContent } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldTooltipContent';
import {
  useConnectionErrorTKey,
  useIsConnectionInProgress,
  useIsConnectionStartField,
} from 'features/nodes/hooks/useFieldConnectionState';
import { useInputFieldIsAddedToForm } from 'features/nodes/hooks/useInputFieldIsAddedToForm';
import { useInputFieldIsConnected } from 'features/nodes/hooks/useInputFieldIsConnected';
import { useInputFieldTemplateTitleOrThrow } from 'features/nodes/hooks/useInputFieldTemplateTitleOrThrow';
import { useInputFieldUserTitleSafe } from 'features/nodes/hooks/useInputFieldUserTitleSafe';
import { fieldLabelChanged } from 'features/nodes/store/nodesSlice';
import { HANDLE_TOOLTIP_OPEN_DELAY, NO_FIT_ON_DOUBLE_CLICK_CLASS } from 'features/nodes/types/constants';
import type { MouseEvent } from 'react';
import { memo, useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLinkBold } from 'react-icons/pi';

const labelSx: SystemStyleObject = {
  p: 0,
  display: 'flex',
  gap: 1,
  alignItems: 'center',
  fontWeight: 'semibold',
  textAlign: 'left',
  color: 'base.300',
  _hover: {
    fontWeight: 'semibold !important',
  },
  '&[data-is-invalid="true"]': {
    color: 'error.300',
  },
  '&[data-is-added-to-form="true"]': {
    color: 'blue.300',
  },
  '&[data-is-disabled="true"]': {
    opacity: 0.5,
  },
};

interface Props {
  nodeId: string;
  fieldName: string;
  isInvalid?: boolean;
  isDragging?: boolean;
}

export const InputFieldTitle = memo((props: Props) => {
  const { nodeId, fieldName, isInvalid, isDragging } = props;
  const inputRef = useRef<HTMLInputElement>(null);
  const label = useInputFieldUserTitleSafe(fieldName);
  const fieldTemplateTitle = useInputFieldTemplateTitleOrThrow(fieldName);
  const { t } = useTranslation();
  const isConnected = useInputFieldIsConnected(fieldName);
  const isAddedToForm = useInputFieldIsAddedToForm(fieldName);
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

  const isDisabled = useMemo(
    () => (isConnectionInProgress && connectionError !== null && !isConnectionStartField) || isConnected,
    [isConnectionInProgress, connectionError, isConnectionStartField, isConnected]
  );

  const onDoubleClick = useCallback(
    (e: MouseEvent<HTMLParagraphElement>) => {
      e.stopPropagation();
      editable.startEditing();
    },
    [editable]
  );

  if (!editable.isEditing) {
    return (
      <Tooltip
        label={<InputFieldTooltipContent nodeId={nodeId} fieldName={fieldName} />}
        openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
        placement="top"
        isDisabled={isDragging}
      >
        <Text
          className={NO_FIT_ON_DOUBLE_CLICK_CLASS}
          sx={labelSx}
          noOfLines={1}
          data-is-invalid={isInvalid}
          data-is-disabled={isDisabled}
          data-is-added-to-form={isAddedToForm}
          onDoubleClick={onDoubleClick}
        >
          {editable.value}
          {isAddedToForm && <Icon as={PiLinkBold} color="blue.200" ml={1} />}
        </Text>
      </Tooltip>
    );
  }

  return (
    <Input
      ref={inputRef}
      variant="outline"
      {...editable.inputProps}
      _focusVisible={{ borderRadius: 'base', h: 'unset', px: 2 }}
    />
  );
});

InputFieldTitle.displayName = 'InputFieldTitle';
