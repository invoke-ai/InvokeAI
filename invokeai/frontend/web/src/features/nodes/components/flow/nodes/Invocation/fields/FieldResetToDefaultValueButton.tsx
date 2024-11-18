import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useFieldInputTemplate } from 'features/nodes/hooks/useFieldInputTemplate';
import { useFieldValue } from 'features/nodes/hooks/useFieldValue';
import { fieldValueReset } from 'features/nodes/store/nodesSlice';
import { isEqual } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';

type Props = {
  nodeId: string;
  fieldName: string;
};

const FieldResetToDefaultValueButton = ({ nodeId, fieldName }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const value = useFieldValue(nodeId, fieldName);
  const fieldTemplate = useFieldInputTemplate(nodeId, fieldName);
  const isDisabled = useMemo(() => {
    return isEqual(value, fieldTemplate.default);
  }, [value, fieldTemplate.default]);
  const onClick = useCallback(() => {
    dispatch(fieldValueReset({ nodeId, fieldName, value: fieldTemplate.default }));
  }, [dispatch, fieldName, fieldTemplate.default, nodeId]);

  return (
    <IconButton
      variant="ghost"
      tooltip={t('nodes.resetToDefaultValue')}
      aria-label={t('nodes.resetToDefaultValue')}
      icon={<PiArrowCounterClockwiseBold />}
      onClick={onClick}
      isDisabled={isDisabled}
      pointerEvents="auto"
      size="xs"
    />
  );
};

export default memo(FieldResetToDefaultValueButton);
