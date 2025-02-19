import { Button, ButtonGroup, Flex, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useGetNodesNeedUpdate } from 'features/nodes/hooks/useGetNodesNeedUpdate';
import { updateAllNodesRequested } from 'features/nodes/store/actions';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiWarningBold } from 'react-icons/pi';

const UpdateNodesButton = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  // const nodesNeedUpdate = useGetNodesNeedUpdate();
  const handleClickUpdateNodes = useCallback(() => {
    dispatch(updateAllNodesRequested());
  }, [dispatch]);

  // if (!nodesNeedUpdate) {
  //   return null;
  // }

  return (
    <Flex flexDir="column">
      <Button
        leftIcon={<PiWarningBold />}
        tooltip={t('nodes.updateAllNodes')}
        aria-label={t('nodes.updateAllNodes')}
        onClick={handleClickUpdateNodes}
        pointerEvents="auto"
        colorScheme="red"
      >
        1 Missing Field
      </Button>
      <ButtonGroup>
        <Button>{`<`}</Button>
        <Button>{`>`}</Button>
      </ButtonGroup>
    </Flex>
  );
};

export default memo(UpdateNodesButton);
