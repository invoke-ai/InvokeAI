import type { SystemStyleObject } from '@chakra-ui/react';
import type { ViewerAutoSwitchMode } from '@workbench/settings/contracts';

import { Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { CheckIcon, SparklesIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

/** The Preview's modes, which the Canvas staging bar shares. */
type AutoSwitchMode = ViewerAutoSwitchMode;

const AUTO_SWITCH_MODES: readonly AutoSwitchMode[] = ['off', 'progress', 'latest'];

/** Both widgets name the modes with the staging bar's keys, so existing translations carry over. */
const MODE_LABEL_KEYS: Record<AutoSwitchMode, string> = {
  latest: 'widgets.canvas.staging.autoSwitchLatest',
  off: 'widgets.canvas.staging.autoSwitchOff',
  progress: 'widgets.canvas.staging.autoSwitchProgress',
};

interface AutoSwitchMenuProps {
  value: AutoSwitchMode;
  onChange: (mode: AutoSwitchMode) => void;
  positioning: { placement: 'bottom-end' | 'top-end' };
  /** `full` shows the current mode beside the icon; `compact` is the icon alone, named by its tooltip. */
  density: 'compact' | 'full';
  /** A host's container rules for the trigger and its mode text, such as the staging bar's narrow collapse. */
  triggerCss?: SystemStyleObject;
  valueCss?: SystemStyleObject;
}

/** What switches the view to new work: a menu of the modes, with the current one checked. */
export const AutoSwitchMenu = ({
  density,
  positioning,
  triggerCss,
  value,
  valueCss,
  onChange,
}: AutoSwitchMenuProps) => {
  const { t } = useTranslation();
  const ids = useTooltipTriggerIds();
  const title = t('widgets.canvas.staging.autoSwitch');
  // Names the current mode on the trigger and its tooltip, so a compact, icon-only trigger still says it.
  const currentLabel = t('widgets.canvas.staging.autoSwitchCurrent', { mode: t(MODE_LABEL_KEYS[value]) });
  const isOff = value === 'off';
  const handleValueChange = useCallback(
    ({ value }: { value: string }) => {
      const mode = AUTO_SWITCH_MODES.find((candidate) => candidate === value);

      if (mode) {
        onChange(mode);
      }
    },
    [onChange]
  );

  return (
    <Menu.Root ids={ids} positioning={positioning}>
      <Tooltip content={currentLabel} ids={ids}>
        <Menu.Trigger asChild>
          <Button
            aria-label={currentLabel}
            css={triggerCss}
            data-auto-switch={value}
            minW="unset"
            px="2"
            variant="ghost"
          >
            {/* Off dims the icon too, so a selection turning Auto-switch off shows on a compact trigger. */}
            <Icon as={SparklesIcon} boxSize="3.5" color={isOff ? 'fg.subtle' : undefined} opacity={isOff ? 0.5 : 1} />
            {density === 'full' ? (
              <Text css={valueCss} fontSize="md">
                {t(MODE_LABEL_KEYS[value])}
              </Text>
            ) : null}
          </Button>
        </Menu.Trigger>
      </Tooltip>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="8rem" py="1">
            <Menu.RadioItemGroup value={value} onValueChange={handleValueChange}>
              <Menu.ItemGroupLabel>{title}</Menu.ItemGroupLabel>
              {AUTO_SWITCH_MODES.map((mode) => (
                <Menu.RadioItem key={mode} value={mode}>
                  <Menu.ItemIndicator>
                    <CheckIcon size={12} />
                  </Menu.ItemIndicator>
                  <Menu.ItemText fontSize="md">{t(MODE_LABEL_KEYS[mode])}</Menu.ItemText>
                </Menu.RadioItem>
              ))}
            </Menu.RadioItemGroup>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
