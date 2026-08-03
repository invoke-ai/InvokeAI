import type { FormEvent, KeyboardEvent } from 'react';

import { chakra, Dialog, Icon, Input, Portal, SimpleGrid, Stack, Text } from '@chakra-ui/react';
import { Button, CloseButton, IconButton } from '@platform/ui/Button';
import { Field } from '@platform/ui/Field';
import { Tooltip } from '@platform/ui/Tooltip';
import { useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { DEFAULT_LAYOUT_PRESET_ICON_ID, layoutPresetIconGroups } from './layoutPresetIcons';

const layoutPresetIconIds = layoutPresetIconGroups.flatMap((group) => group.options.map((option) => option.id));

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
 * Name and icon for a custom layout preset.
 *
 * Custom presets sit in the same strip as the built-ins and collapse to
 * icon-only below 1280px, so an icon is not decoration — without one every
 * custom preset would be an identical anonymous square at that width. The
 * picker offers a curated set for exactly that reason: the job is to tell your
 * own layouts apart at a glance, not to browse an icon library.
 */
export const LayoutPresetDialog = ({
  iconId: initialIconId,
  isOpen,
  name: initialName,
  onClose,
  onSubmit,
  submitLabel,
  title,
}: {
  iconId?: string;
  isOpen: boolean;
  name: string;
  onClose: () => void;
  onSubmit: (value: { iconId: string; name: string }) => void;
  submitLabel: string;
  title: string;
}) => {
  const { t } = useTranslation();
  const [name, setName] = useState(initialName);
  const [iconId, setIconId] = useState(initialIconId ?? DEFAULT_LAYOUT_PRESET_ICON_ID);
  const nameRef = useRef<HTMLInputElement>(null);

  // Without this the focus trap lands on the header's close button, so opening
  // the dialog and typing does nothing. `initialFocusEl` rather than `autoFocus`
  // so the trap itself does the focusing, at the point it is ready to.
  const initialFocusEl = useCallback(() => nameRef.current, []);

  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        onClose();
      }
    },
    [onClose]
  );
  const handleNameChange = useCallback(
    (event: { currentTarget: { value: string } }) => setName(event.currentTarget.value),
    []
  );
  const handleSubmit = useCallback(
    (event: FormEvent) => {
      event.preventDefault();
      const trimmed = name.trim();

      if (!trimmed) {
        onClose();

        return;
      }

      onSubmit({ iconId, name: trimmed });
      onClose();
    },
    [iconId, name, onClose, onSubmit]
  );
  const handleIconKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      const nextIconId = getNextIconId(iconId, event.key);

      if (!nextIconId) {
        return;
      }

      event.preventDefault();
      setIconId(nextIconId);
      const nextButton = [...event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="radio"]')].find(
        (button) => button.dataset.iconId === nextIconId
      );
      nextButton?.focus();
    },
    [iconId]
  );

  return (
    <Dialog.Root initialFocusEl={initialFocusEl} open={isOpen} lazyMount unmountOnExit onOpenChange={handleOpenChange}>
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content asChild maxW="26rem">
            <chakra.form onSubmit={handleSubmit}>
              <Dialog.Header>
                <Dialog.Title>{title}</Dialog.Title>
                <Dialog.CloseTrigger asChild>
                  <CloseButton size="sm" type="button" />
                </Dialog.CloseTrigger>
              </Dialog.Header>
              <Dialog.Body>
                <Stack gap="4">
                  <Field label={t('topbar.presets.name')}>
                    <Input
                      ref={nameRef}
                      autoComplete="off"
                      name="layout-preset-name"
                      size="sm"
                      value={name}
                      onChange={handleNameChange}
                    />
                  </Field>
                  <Stack
                    aria-label={t('topbar.presets.iconPicker')}
                    gap="3"
                    role="radiogroup"
                    tabIndex={-1}
                    onKeyDown={handleIconKeyDown}
                  >
                    <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
                      {t('topbar.presets.icon')}
                    </Text>
                    {layoutPresetIconGroups.map((group) => (
                      <Stack key={group.label} gap="1.5">
                        <Text color="fg.muted" fontSize="2xs">
                          {group.label}
                        </Text>
                        <SimpleGrid columns={8} gap="1">
                          {group.options.map((entry) => (
                            <IconOption
                              key={entry.id}
                              icon={entry.icon}
                              iconId={entry.id}
                              isSelected={entry.id === iconId}
                              label={entry.label}
                              onSelect={setIconId}
                            />
                          ))}
                        </SimpleGrid>
                      </Stack>
                    ))}
                  </Stack>
                </Stack>
              </Dialog.Body>
              <Dialog.Footer>
                <Button size="sm" type="button" variant="outline" onClick={onClose}>
                  {t('common.cancel')}
                </Button>
                <Button disabled={name.trim().length === 0} size="sm" type="submit">
                  {submitLabel}
                </Button>
              </Dialog.Footer>
            </chakra.form>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
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
    <Tooltip content={label} showArrow>
      <IconButton
        aria-checked={isSelected}
        aria-label={label}
        colorPalette={isSelected ? 'accent' : undefined}
        data-icon-id={iconId}
        role="radio"
        size="sm"
        tabIndex={isSelected ? 0 : -1}
        type="button"
        variant={isSelected ? 'solid' : 'ghost'}
        onClick={handleClick}
      >
        <Icon as={icon} />
      </IconButton>
    </Tooltip>
  );
};
