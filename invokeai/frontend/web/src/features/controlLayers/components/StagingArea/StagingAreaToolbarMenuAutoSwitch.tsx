/* eslint-disable i18next/no-literal-string */
import { MenuItemOption, MenuOptionGroup } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasSessionContext, zAutoSwitchMode } from 'features/controlLayers/components/SimpleSession/context';
import { memo, useCallback } from 'react';

export const StagingAreaToolbarMenuAutoSwitch = memo(() => {
  const ctx = useCanvasSessionContext();
  const autoSwitch = useStore(ctx.$autoSwitch);

  const onChange = useCallback(
    (val: string | string[]) => {
      const newAutoSwitch = zAutoSwitchMode.parse(val);
      ctx.$autoSwitch.set(newAutoSwitch);
    },
    [ctx.$autoSwitch]
  );

  return (
    <MenuOptionGroup value={autoSwitch} onChange={onChange} title="Auto Switch" type="radio">
      <MenuItemOption value="off" closeOnSelect={false}>
        Off
      </MenuItemOption>
      <MenuItemOption value="first_progress" closeOnSelect={false}>
        First Progress
      </MenuItemOption>
      <MenuItemOption value="completed" closeOnSelect={false}>
        Completed
      </MenuItemOption>
    </MenuOptionGroup>
  );
});

StagingAreaToolbarMenuAutoSwitch.displayName = 'StagingAreaToolbarMenuAutoSwitch';
