import { MenuItemOption, MenuOptionGroup } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { memo, useCallback } from 'react';

export const StagingAreaToolbarMenuAutoSwitch = memo(() => {
  const ctx = useCanvasSessionContext();
  const autoSwitch = useStore(ctx.$autoSwitch);

  const onChange = useCallback(
    (val: string | string[]) => {
      ctx.$autoSwitch.set(val === 'on');
    },
    [ctx.$autoSwitch]
  );

  return (
    <MenuOptionGroup value={autoSwitch ? 'on' : 'off'} onChange={onChange} title="Auto Switch" type="radio">
      <MenuItemOption value="off" closeOnSelect={false}>
        Off
      </MenuItemOption>
      <MenuItemOption value="on" closeOnSelect={false}>
        On
      </MenuItemOption>
    </MenuOptionGroup>
  );
});

StagingAreaToolbarMenuAutoSwitch.displayName = 'StagingAreaToolbarMenuAutoSwitch';
