import { Flex } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { nodeOpacityChanged } from 'features/nodes/store/nodesSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const NodeOpacitySlider = () => {
  const dispatch = useAppDispatch();
  const nodeOpacity = useAppSelector((state) => state.nodes.nodeOpacity);
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(nodeOpacityChanged(v));
    },
    [dispatch]
  );

  return (
    <Flex alignItems="center">
      <InvSlider
        aria-label={t('nodes.nodeOpacity')}
        value={nodeOpacity}
        min={0.5}
        max={1}
        step={0.01}
        onChange={handleChange}
        orientation="vertical"
        defaultValue={30}
        h="calc(100% - 0.5rem)"
      />
    </Flex>
  );
};

export default memo(NodeOpacitySlider);
