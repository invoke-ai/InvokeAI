import { Box, Menu } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { MenuActionItem, MenuContent, MenuHeadingGroup } from '@platform/ui/Menu';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { useCallback, useRef, useState, type MouseEvent } from 'react';
import { useTranslation } from 'react-i18next';

import type { SettingLayer } from './contracts';

const MENU_POSITIONING = { placement: 'bottom-start', strategy: 'fixed', hideWhenDetached: true } as const;
/** The control of a setting row, which takes focus when an action removes the indicator under it. */
const ROW_CONTROL = 'input:not([type="hidden"]), button:not([data-setting-layer]), [role="slider"]';

const preventLabelActivation = (event: MouseEvent) => event.preventDefault();

const Dot = () => <Box as="span" boxSize="1.5" flexShrink={0} borderRadius="full" bg="accent.fg" />;

/**
 * The dot after a setting's label: the row holds a value of its own, either this project's pin or a change from the
 * built-in. A 24px button opening a menu of its actions; a change with no reset (the row has its own way back) is a
 * plain glyph that only explains itself.
 */
export const SettingLayerIndicator = ({ label, layer }: { label: string; layer: SettingLayer }) => {
  const { t } = useTranslation();
  const ids = useTooltipTriggerIds();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const handleOpenChange = useCallback((details: { open: boolean }) => setIsMenuOpen(details.open), []);
  const trigger = useRef<HTMLButtonElement>(null);
  // Every action returns the row to following its default, which removes this indicator and the trigger the menu
  // would restore focus to: focus goes to the row's control instead (`data-setting-control` marks the row).
  const select = useCallback(
    (action: () => void) => () => {
      const row = trigger.current?.closest('[data-setting-control]');
      action();
      requestAnimationFrame(() => {
        if (!trigger.current?.isConnected) {
          row?.querySelector<HTMLElement>(ROW_CONTROL)?.focus();
        }
      });
    },
    []
  );
  const pinned = layer.state === 'pinned';
  // The tooltip sits beside the label, so it stays short; the accessible name says which setting it belongs to, and
  // `aria-haspopup` announces the menu, so the name does not repeat "options".
  const accessibleName = t(pinned ? 'settingsDialog.layer.pinnedNamed' : 'settingsDialog.layer.changedNamed', {
    setting: label,
  });
  if (layer.state === 'default' && !layer.onResetToBuiltIn) {
    return (
      <Tooltip content={t('settingsDialog.layer.changed')}>
        <Box as="span" role="img" aria-label={accessibleName} display="inline-flex" flexShrink={0}>
          <Dot />
        </Box>
      </Tooltip>
    );
  }

  return (
    // Inline with fixed positioning, as the setting Select is: fixed escapes the gear popover's scroll area, and staying
    // inside the modal dialog's DOM keeps its focus scope from pulling focus out of the menu and dismissing it. Mounted
    // up front: a menu mounted on first open let the dialog's focus scope claim focus for a frame before the menu did.
    <Menu.Root ids={ids} positioning={MENU_POSITIONING} onOpenChange={handleOpenChange}>
      {/* Quiet while the menu is open: it opens below the dot, where the tooltip would cover its first item. */}
      <Tooltip
        content={t(pinned ? 'settingsDialog.layer.pinnedOptions' : 'settingsDialog.layer.changedOptions')}
        disabled={isMenuOpen}
        ids={ids}
        keepsRootWhenDisabled
      >
        <Menu.Trigger asChild>
          <IconButton
            ref={trigger}
            aria-label={accessibleName}
            data-setting-layer=""
            // A 24px target around a 6px glyph; the negative margins keep the label row at its text height.
            boxSize="6"
            minW="6"
            my="-1.5"
            ms="-1.5"
            size="xs"
            variant="ghost"
            flexShrink={0}
          >
            <Dot />
          </IconButton>
        </Menu.Trigger>
      </Tooltip>
      {/* Inline, the menu sits inside a boolean row's <label>: a click on an item would also reach the label and toggle
          the switch. Preventing the click's default stops that activation; the item has already run its action. */}
      <Menu.Positioner onClick={preventLabelActivation}>
        <MenuContent minW="14rem" maxW="20rem">
          {layer.state === 'pinned' ? (
            <MenuHeadingGroup label={t('settingsDialog.layer.thisProjectOnly')}>
              <MenuActionItem
                value="use-as-default"
                label={t('settingsDialog.layer.useAsDefault')}
                hint={t('settingsDialog.layer.useAsDefaultHint')}
                onSelect={select(layer.onUseAsDefault)}
              />
              <MenuActionItem
                value="follow-default"
                label={t('settingsDialog.layer.followDefault')}
                hint={layer.defaultValueLabel}
                onSelect={select(layer.onFollowDefault)}
              />
            </MenuHeadingGroup>
          ) : layer.onResetToBuiltIn ? (
            <MenuActionItem
              value="reset-to-built-in"
              label={t('settingsDialog.layer.resetToBuiltIn')}
              onSelect={select(layer.onResetToBuiltIn)}
            />
          ) : null}
        </MenuContent>
      </Menu.Positioner>
    </Menu.Root>
  );
};
