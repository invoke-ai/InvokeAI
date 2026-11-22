import type { KeyboardEvent } from 'react';

import { Icon, SimpleGrid, Stack } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { layoutPresetIconGroups } from './layoutPresetIcons';

const layoutPresetIconOptions = layoutPresetIconGroups.flatMap((group) => group.options);
const layoutPresetIconIds = layoutPresetIconOptions.map((option) => option.id);

const getNextIconId = (currentIconId: string, key: string): string | null => {
  if (key === 'Home') {
    return layoutPresetIconIds[0] ?? null;
  }
  if (key === 'End') {
    return layoutPresetIconIds.at(-1) ?? null;
  }

  const direction = key === 'ArrowRight' || key === 'ArrowDown' ? 1 : key === 'ArrowLeft' || key === 'ArrowUp' ? -1 : 0;
  if (direction === 0) {
    return null;
  }

  const currentIndex = layoutPresetIconIds.indexOf(currentIconId);
  const nextIndex = (Math.max(currentIndex, 0) + direction + layoutPresetIconIds.length) % layoutPresetIconIds.length;
  return layoutPresetIconIds[nextIndex] ?? null;
};

/**
 * Workspaces become icon-only below 1280px; a curated picker keeps them distinguishable without a full icon search.
 * One radio group: arrow keys, Home and End move the choice and focus together. `iconId` is a known id
 * (`getInitialLayoutPresetIconId`), so exactly one option is in the tab order.
 */
export const LayoutPresetIconPicker = ({
  iconId,
  onChange,
}: {
  iconId: string;
  onChange: (iconId: string) => void;
}) => {
  const { t } = useTranslation();
  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      const nextIconId = getNextIconId(iconId, event.key);

      if (!nextIconId) {
        return;
      }

      event.preventDefault();
      onChange(nextIconId);
      [...event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="radio"]')]
        .find((button) => button.dataset.iconId === nextIconId)
        ?.focus();
    },
    [iconId, onChange]
  );

  return (
    <Stack aria-label={t('topbar.presets.iconPicker')} role="radiogroup" tabIndex={-1} onKeyDown={handleKeyDown}>
      <SimpleGrid columns={8} gap="1">
        {layoutPresetIconOptions.map((entry) => (
          <IconOption
            key={entry.id}
            icon={entry.icon}
            iconId={entry.id}
            isSelected={entry.id === iconId}
            label={entry.label}
            onSelect={onChange}
          />
        ))}
      </SimpleGrid>
    </Stack>
  );
};

const IconOption = ({
  icon,
  iconId,
  isSelected,
  label,
  onSelect,
}: {
  icon: (typeof layoutPresetIconGroups)[number]['options'][number]['icon'];
  iconId: string;
  isSelected: boolean;
  label: string;
  onSelect: (iconId: string) => void;
}) => {
  const handleClick = useCallback(() => onSelect(iconId), [iconId, onSelect]);

  return (
    <IconButton
      aria-checked={isSelected}
      aria-label={label}
      colorPalette={isSelected ? 'accent' : undefined}
      data-icon-id={iconId}
      role="radio"
      size="lg"
      tabIndex={isSelected ? 0 : -1}
      tooltip={label}
      type="button"
      variant={isSelected ? 'solid' : 'ghost'}
      onClick={handleClick}
    >
      <Icon as={icon} />
    </IconButton>
  );
};
