import type { SystemStyleObject } from '@chakra-ui/react';
import type { ViewerAutoSwitchMode } from '@workbench/settings/contracts';

import { Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { ActivityIcon, CheckIcon, GalleryHorizontalEndIcon, LocateOffIcon, type LucideIcon } from 'lucide-react';
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

/**
 * Each mode's shape says what it does, on the trigger and its menu item alike: Off is a crossed-out follow target (the
 * view stays put), In progress a live trace (it follows a generation as it streams), Newest the front of a stack (it
 * jumps to each finished result). Pin is not used for Off: the Preview filmstrip already marks a pinned session with it.
 */
const MODE_ICONS: Record<AutoSwitchMode, LucideIcon> = {
  latest: GalleryHorizontalEndIcon,
  off: LocateOffIcon,
  progress: ActivityIcon,
};

/**
 * Trigger styles per host, so it matches the controls beside it: a widget header's small, muted ghost buttons (24px,
 * 14px icons), or a floating toolbar's working-size ones (28px, 16px icons). The size recipe sets the icon, padding and
 * text, as it does for those neighbours.
 */
const TRIGGER_CHROME = {
  header: { button: { color: 'fg.muted', size: 'sm' } },
  toolbar: { button: { size: 'md' } },
} as const;

interface AutoSwitchMenuProps {
  value: AutoSwitchMode;
  onChange: (mode: AutoSwitchMode) => void;
  positioning: { placement: 'bottom-end' | 'top-end' };
  /** `full` shows the current mode beside the icon; `compact` is the icon alone, named by its tooltip. */
  density: 'compact' | 'full';
  /** Which neighbouring controls the trigger matches; `toolbar` by default. */
  chrome?: keyof typeof TRIGGER_CHROME;
  /** A host's container rules for the trigger and its mode text, such as the staging bar's narrow collapse. */
  triggerCss?: SystemStyleObject;
  valueCss?: SystemStyleObject;
}

/** What switches the view to new work: a menu of the modes, with the current one checked. */
export const AutoSwitchMenu = ({
  chrome = 'toolbar',
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
  const ModeIcon = MODE_ICONS[value];
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
            // Icon alone, it is square like the icon buttons beside it.
            px={density === 'compact' ? '0' : undefined}
            variant="ghost"
            {...TRIGGER_CHROME[chrome].button}
          >
            <ModeIcon aria-hidden />
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
                  <Icon aria-hidden as={MODE_ICONS[mode]} boxSize="3.5" color="fg.subtle" flexShrink={0} />
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
