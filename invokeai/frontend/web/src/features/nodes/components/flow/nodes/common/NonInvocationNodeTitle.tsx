import { Flex, Input, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEditable } from 'common/hooks/useEditable';
import { nodeLabelChanged } from 'features/nodes/store/nodesSlice';
import { selectNodes } from 'features/nodes/store/selectors';
import { NO_DRAG_CLASS, NO_FIT_ON_DOUBLE_CLICK_CLASS } from 'features/nodes/types/constants';
import { memo, useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  nodeId: string;
  title: string;
};

const NonInvocationNodeTitle = ({ nodeId, title }: Props) => {
  const dispatch = useAppDispatch();
  const selectNodeLabel = useMemo(
    () =>
      createSelector(selectNodes, (nodes) => {
        const node = nodes.find((n) => n.id === nodeId);
        return node?.data?.label ?? '';
      }),
    [nodeId]
  );
  const label = useAppSelector(selectNodeLabel);
  const { t } = useTranslation();
  const inputRef = useRef<HTMLInputElement>(null);

  const onChange = useCallback(
    (label: string) => {
      dispatch(nodeLabelChanged({ nodeId, label }));
    },
    [dispatch, nodeId]
  );

  const editable = useEditable({
    value: label || title || t('nodes.problemSettingTitle'),
    defaultValue: title || t('nodes.problemSettingTitle'),
    onChange,
    inputRef,
  });

  return (
    <Flex overflow="hidden" w="full" h="full" alignItems="center" justifyContent="center">
      {!editable.isEditing && (
        <Text
          className={NO_FIT_ON_DOUBLE_CLICK_CLASS}
          fontWeight="semibold"
          color="base.200"
          onDoubleClick={editable.startEditing}
          noOfLines={1}
        >
          {editable.value}
        </Text>
      )}
      {editable.isEditing && (
        <Input
          className={NO_DRAG_CLASS}
          ref={inputRef}
          {...editable.inputProps}
          variant="outline"
          _focusVisible={{ borderRadius: 'base', h: 'unset' }}
        />
      )}
    </Flex>
  );
};

export default memo(NonInvocationNodeTitle);
