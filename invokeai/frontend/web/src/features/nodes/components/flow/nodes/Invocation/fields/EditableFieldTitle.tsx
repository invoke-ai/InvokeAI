import type { SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Editable,
  EditableInput,
  EditablePreview,
  Flex,
  forwardRef,
  Tooltip,
  useEditableControls,
} from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useFieldLabel } from 'features/nodes/hooks/useFieldLabel';
import { useFieldTemplateTitle } from 'features/nodes/hooks/useFieldTemplateTitle';
import { fieldLabelChanged } from 'features/nodes/store/nodesSlice';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import type { MouseEvent } from 'react';
import { memo, useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

import FieldTooltipContent from './FieldTooltipContent';

interface Props {
  nodeId: string;
  fieldName: string;
  kind: 'inputs' | 'outputs';
  isMissingInput?: boolean;
  withTooltip?: boolean;
}

const EditableFieldTitle = forwardRef((props: Props, ref) => {
  const { nodeId, fieldName, kind, isMissingInput = false, withTooltip = false } = props;
  const label = useFieldLabel(nodeId, fieldName);
  const fieldTemplateTitle = useFieldTemplateTitle(nodeId, fieldName, kind);
  const { t } = useTranslation();

  const dispatch = useAppDispatch();
  const [localTitle, setLocalTitle] = useState(label || fieldTemplateTitle || t('nodes.unknownField'));

  const handleSubmit = useCallback(
    async (newTitle: string) => {
      if (newTitle && (newTitle === label || newTitle === fieldTemplateTitle)) {
        return;
      }
      setLocalTitle(newTitle || fieldTemplateTitle || t('nodes.unknownField'));
      dispatch(fieldLabelChanged({ nodeId, fieldName, label: newTitle }));
    },
    [label, fieldTemplateTitle, dispatch, nodeId, fieldName, t]
  );

  const handleChange = useCallback((newTitle: string) => {
    setLocalTitle(newTitle);
  }, []);

  useEffect(() => {
    // Another component may change the title; sync local title with global state
    setLocalTitle(label || fieldTemplateTitle || t('nodes.unknownField'));
  }, [label, fieldTemplateTitle, t]);

  return (
    <Tooltip
      label={withTooltip ? <FieldTooltipContent nodeId={nodeId} fieldName={fieldName} kind="inputs" /> : undefined}
      openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
    >
      <Editable
        value={localTitle}
        onChange={handleChange}
        onSubmit={handleSubmit}
        as={Flex}
        ref={ref}
        position="relative"
        overflow="hidden"
        alignItems="center"
        justifyContent="flex-start"
        gap={1}
        w="full"
      >
        <EditablePreview
          fontWeight="semibold"
          sx={editablePreviewStyles}
          noOfLines={1}
          color={isMissingInput ? 'error.300' : 'base.300'}
        />
        <EditableInput className="nodrag" sx={editableInputStyles} />
        <EditableControls />
      </Editable>
    </Tooltip>
  );
});

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
const editablePreviewStyles: SystemStyleObject = {
  p: 0,
  textAlign: 'left',
  _hover: {
    fontWeight: 'semibold !important',
  },
};

export default memo(EditableFieldTitle);

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
    <Flex onClick={handleClick} position="absolute" w="full" h="full" top={0} insetInlineStart={0} cursor="text" />
  );
});

EditableControls.displayName = 'EditableControls';
