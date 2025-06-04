/* eslint-disable i18next/no-literal-string */
import { Button, Flex, FormControl, FormLabel, Spacer, Switch, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { canvasSessionStarted } from 'features/controlLayers/store/canvasStagingAreaSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';

export const StagingAreaHeader = memo(() => {
  const ctx = useCanvasSessionContext();
  const autoSwitch = useStore(ctx.$autoSwitch);
  const dispatch = useAppDispatch();

  const startOver = useCallback(() => {
    dispatch(canvasSessionStarted({ sessionType: 'simple' }));
  }, [dispatch]);

  const onChangeAutoSwitch = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      ctx.$autoSwitch.set(e.target.checked);
    },
    [ctx.$autoSwitch]
  );

  return (
    <Flex gap={2} w="full" alignItems="center">
      <Text fontSize="lg" fontWeight="bold">
        Generations
      </Text>
      <Spacer />
      <FormControl w="min-content">
        <FormLabel m={0}>Auto-switch</FormLabel>
        <Switch size="sm" isChecked={autoSwitch} onChange={onChangeAutoSwitch} />
      </FormControl>
      <Button size="sm" variant="ghost" onClick={startOver}>
        Start Over
      </Button>
    </Flex>
  );
});
StagingAreaHeader.displayName = 'StagingAreaHeader';
