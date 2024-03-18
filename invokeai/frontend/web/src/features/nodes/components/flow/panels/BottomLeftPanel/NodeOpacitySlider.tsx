import { CompositeSlider, Flex } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { nodeOpacityChanged } from 'features/nodes/store/nodesSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const NodeOpacitySlider = () => {
  const dispatch = useAppDispatch();
  const nodeOpacity = useAppSelector((s) => s.nodes.nodeOpacity);
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(nodeOpacityChanged(v));
    },
    [dispatch]
  );

  return (
    <Flex alignItems="center">
      <CompositeSlider
        aria-label={t('nodes.nodeOpacity')}
        value={nodeOpacity}
        defaultValue={1}
        min={0.5}
        max={1}
        step={0.01}
        onChange={handleChange}
        orientation="vertical"
        h="calc(100% - 0.5rem)"
      />
    </Flex>
  );
};

export default memo(NodeOpacitySlider);
