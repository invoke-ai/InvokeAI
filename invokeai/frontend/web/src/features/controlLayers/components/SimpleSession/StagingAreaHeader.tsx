/* eslint-disable i18next/no-literal-string */
import { Divider, Flex, FormControl, FormLabel, Spacer, Switch, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { StartOverButton } from 'features/controlLayers/components/StartOverButton';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';

export const StagingAreaHeader = memo(() => {
  const ctx = useCanvasSessionContext();
  const autoSwitch = useStore(ctx.$autoSwitch);

  const onChangeAutoSwitch = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      ctx.$autoSwitch.set(e.target.checked);
    },
    [ctx.$autoSwitch]
  );

  return (
    <Flex gap={2} w="full" alignItems="center" px={2}>
      <Text fontSize="lg" fontWeight="bold">
        Staging Area
      </Text>
      <Spacer />
      <FormControl w="min-content" me={2}>
        <FormLabel m={0}>Auto-switch</FormLabel>
        <Switch size="sm" isChecked={autoSwitch} onChange={onChangeAutoSwitch} />
      </FormControl>
      <Divider orientation="vertical" />
      <StartOverButton />
    </Flex>
  );
});
StagingAreaHeader.displayName = 'StagingAreaHeader';
