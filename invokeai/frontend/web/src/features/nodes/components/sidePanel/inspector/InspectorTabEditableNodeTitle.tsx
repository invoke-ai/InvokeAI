import { Flex, Input, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEditable } from 'common/hooks/useEditable';
import { useNodeLabel } from 'features/nodes/hooks/useNodeLabel';
import { useNodeTemplateTitle } from 'features/nodes/hooks/useNodeTemplateTitle';
import { nodeLabelChanged } from 'features/nodes/store/nodesSlice';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  nodeId: string;
  title?: string;
};

const InspectorTabEditableNodeTitle = ({ nodeId, title }: Props) => {
  const dispatch = useAppDispatch();
  const label = useNodeLabel(nodeId);
  const templateTitle = useNodeTemplateTitle(nodeId);
  const { t } = useTranslation();
  const inputRef = useRef<HTMLInputElement>(null);
  const onChange = useCallback(
    (label: string) => {
      dispatch(nodeLabelChanged({ nodeId, label }));
    },
    [dispatch, nodeId]
  );
  const editable = useEditable({
    value: label || title || templateTitle || t('nodes.problemSettingTitle'),
    defaultValue: title || templateTitle || t('nodes.problemSettingTitle'),
    onChange,
    inputRef,
  });

  return (
    <Flex w="full" alignItems="center">
      {!editable.isEditing && (
        <Text size="sm" fontWeight="semibold" userSelect="none" onDoubleClick={editable.startEditing}>
          {editable.value}
        </Text>
      )}
      {editable.isEditing && (
        <Input
          ref={inputRef}
          {...editable.inputProps}
          variant="outline"
          _focusVisible={{ borderRadius: 'base', h: 'unset' }}
        />
      )}
    </Flex>
  );
};

export default memo(InspectorTabEditableNodeTitle);
