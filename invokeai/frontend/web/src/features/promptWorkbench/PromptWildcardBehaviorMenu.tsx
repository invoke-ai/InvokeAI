import { IconButton, Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import type { MouseEvent, ReactElement } from 'react';
import { memo, useCallback } from 'react';
import { PiDiceFiveBold, PiPushPinSimpleBold, PiRepeatBold, PiSquaresFourBold, PiTrashSimpleBold, PiWarningBold } from 'react-icons/pi';

import type { WildcardBehaviorAction, WildcardBehaviorIconType } from './occurrences';

type PromptWildcardBehaviorMenuProps = {
  ariaLabel: string;
  tooltip: string;
  iconType: WildcardBehaviorIconType;
  isActionable: boolean;
  canPickFixedValue: boolean;
  includeRemove: boolean;
  onAction: (action: WildcardBehaviorAction) => void;
  onOpen?: () => void;
  onClose?: () => void;
};

const BUTTON_SIZE = 7;

export const PromptWildcardBehaviorMenu = memo(
  ({
    ariaLabel,
    tooltip,
    iconType,
    isActionable,
    canPickFixedValue,
    includeRemove,
    onAction,
    onOpen,
    onClose,
  }: PromptWildcardBehaviorMenuProps) => {
    const onButtonMouseDown = useCallback((e: MouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
    }, []);

    const onRandom = useCallback(() => {
      onAction('random');
    }, [onAction]);

    const onCycle = useCallback(() => {
      onAction('cycle');
    }, [onAction]);

    const onFixed = useCallback(() => {
      onAction('fixed');
    }, [onAction]);

    const onRemove = useCallback(() => {
      onAction('remove');
    }, [onAction]);

    return (
      <Menu onOpen={onOpen} onClose={onClose}>
        <MenuButton
          as={IconButton}
          aria-label={ariaLabel}
          tooltip={tooltip}
          size="xs"
          variant="ghost"
          minW={BUTTON_SIZE}
          h={BUTTON_SIZE}
          icon={getBehaviorMenuIcon(iconType)}
          onMouseDown={onButtonMouseDown}
        />
        <MenuList>
          {isActionable && (
            <>
              <MenuItem
                icon={<PiDiceFiveBold />}
                onClick={onRandom}
                title="Use this wildcard as a random token. Random cadence is controlled by the prompt."
              >
                Random wildcard
              </MenuItem>
              <MenuItem icon={<PiRepeatBold />} onClick={onCycle} title="Cycles through values across generated outputs.">
                Cycle through values
              </MenuItem>
              <MenuItem icon={<PiPushPinSimpleBold />} onClick={onFixed} isDisabled={!canPickFixedValue}>
                Pick fixed value
              </MenuItem>
            </>
          )}
          {includeRemove && (
            <MenuItem icon={<PiTrashSimpleBold />} color="error.300" onClick={onRemove}>
              Remove
            </MenuItem>
          )}
        </MenuList>
      </Menu>
    );
  }
);

PromptWildcardBehaviorMenu.displayName = 'PromptWildcardBehaviorMenu';

const getBehaviorMenuIcon = (iconType: WildcardBehaviorIconType): ReactElement => {
  switch (iconType) {
    case 'random':
      return <PiDiceFiveBold />;
    case 'cycle':
      return <PiRepeatBold />;
    case 'all':
      return <PiSquaresFourBold />;
    case 'warning':
      return <PiWarningBold />;
  }
};
