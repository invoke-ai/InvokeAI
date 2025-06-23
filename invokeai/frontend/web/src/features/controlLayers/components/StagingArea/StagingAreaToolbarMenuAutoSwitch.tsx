import { MenuItemOption, MenuOptionGroup } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { isAutoSwitchMode, useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { memo, useCallback } from 'react';
import { assert } from 'tsafe';

export const StagingAreaToolbarMenuAutoSwitch = memo(() => {
  const ctx = useCanvasSessionContext();
  const autoSwitch = useStore(ctx.$autoSwitch);

  const onChange = useCallback(
    (val: string | string[]) => {
      assert(isAutoSwitchMode(val));
      ctx.$autoSwitch.set(val);
    },
    [ctx.$autoSwitch]
  );

  return (
    <MenuOptionGroup value={autoSwitch} onChange={onChange} title="Auto-Switch" type="radio">
      <MenuItemOption value="off" closeOnSelect={false}>
        Off
      </MenuItemOption>
      <MenuItemOption value="switch_on_start" closeOnSelect={false}>
        Switch on Start
      </MenuItemOption>
      <MenuItemOption value="switch_on_finish" closeOnSelect={false}>
        Switch on Finish
      </MenuItemOption>
    </MenuOptionGroup>
  );
});

StagingAreaToolbarMenuAutoSwitch.displayName = 'StagingAreaToolbarMenuAutoSwitch';
