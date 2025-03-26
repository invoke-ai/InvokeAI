import { Flex, Input, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEditable } from 'common/hooks/useEditable';
import { useBatchGroupColorToken } from 'features/nodes/hooks/useBatchGroupColorToken';
import { useBatchGroupId } from 'features/nodes/hooks/useBatchGroupId';
import { useNodeTemplateTitleSafe } from 'features/nodes/hooks/useNodeTemplateTitleSafe';
import { useNodeUserTitleSafe } from 'features/nodes/hooks/useNodeUserTitleSafe';
import { nodeLabelChanged } from 'features/nodes/store/nodesSlice';
import { NO_FIT_ON_DOUBLE_CLICK_CLASS } from 'features/nodes/types/constants';
import { memo, useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  nodeId: string;
  title?: string;
};

const NodeTitle = ({ nodeId, title }: Props) => {
  const dispatch = useAppDispatch();
  const label = useNodeUserTitleSafe(nodeId);
  const batchGroupId = useBatchGroupId(nodeId);
  const batchGroupColorToken = useBatchGroupColorToken(batchGroupId);
  const templateTitle = useNodeTemplateTitleSafe(nodeId);
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

  const titleWithBatchGroupId = useMemo(() => {
    if (!batchGroupId) {
      return editable.value;
    }
    if (batchGroupId === 'None') {
      return `${editable.value} (${t('nodes.noBatchGroup')})`;
    }
    return `${editable.value} (${batchGroupId})`;
  }, [batchGroupId, editable.value, t]);

  return (
    <Flex overflow="hidden" w="full" h="full" alignItems="center" justifyContent="center">
      {!editable.isEditing && (
        <Text
          className={NO_FIT_ON_DOUBLE_CLICK_CLASS}
          fontWeight="semibold"
          color={batchGroupColorToken}
          onDoubleClick={editable.startEditing}
          noOfLines={1}
        >
          {titleWithBatchGroupId}
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

export default memo(NodeTitle);
