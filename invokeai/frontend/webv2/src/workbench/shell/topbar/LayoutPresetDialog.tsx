import type { GraphWidgetSource } from '@workbench/graphWidgets';
import type { InvocationSourceId, ResultDestination } from '@workbench/invocationContracts';
import type { LayoutPresetRoute } from '@workbench/layoutContracts';
import type { FormEvent, KeyboardEvent } from 'react';

import {
  chakra,
  createListCollection,
  Dialog,
  HStack,
  Icon,
  Input,
  Portal,
  SegmentGroup,
  SimpleGrid,
  Stack,
  Text,
} from '@chakra-ui/react';
import { Button, CloseButton, IconButton } from '@platform/ui/Button';
import { Field } from '@platform/ui/Field';
import { Select } from '@platform/ui/Select';
import { Tooltip } from '@platform/ui/Tooltip';
import { getNaturalDestination } from '@workbench/graphWidgets';
import { WidgetIcon } from '@workbench/iconResolver';
import { getDestinationLabel, resultDestinations } from '@workbench/invocation';
import { getWidgetById } from '@workbench/widgetRegistry';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { getInitialLayoutPresetRoute } from './layoutPresetDialogModel';
import { DEFAULT_LAYOUT_PRESET_ICON_ID, layoutPresetIconGroups } from './layoutPresetIcons';

const layoutPresetIconIds = layoutPresetIconGroups.flatMap((group) => group.options.map((option) => option.id));
const destinationWidgetTypeIds: Record<ResultDestination, 'canvas' | 'gallery'> = {
  canvas: 'canvas',
  gallery: 'gallery',
};

type SourceSelectItem = GraphWidgetSource & { value: InvocationSourceId };

const renderSourceOption = (source: SourceSelectItem) => (
  <HStack gap="1.5">
    <WidgetIcon boxSize="3.5" icon={getWidgetById(source.typeId)?.manifest.icon} />
    <Text as="span">{source.label}</Text>
  </HStack>
);

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
 * Identity and default routing for a layout preset.
 *
 * Presets collapse to icon-only below 1280px, so the icon is not decoration.
 * The curated picker keeps the task focused on telling layouts apart at a
 * glance instead of turning it into an icon-library search.
 */
export const LayoutPresetDialog = ({
  defaultRoute: initialDefaultRoute,
  iconId: initialIconId,
  isOpen,
  name: initialName,
  onClose,
  onSubmit,
  sourceOptions,
  submitLabel,
  title,
}: {
  defaultRoute?: LayoutPresetRoute;
  iconId?: string;
  isOpen: boolean;
  name: string;
  onClose: () => void;
  onSubmit: (value: { defaultRoute: LayoutPresetRoute | null; iconId: string; name: string }) => void;
  sourceOptions: readonly GraphWidgetSource[];
  submitLabel: string;
  title: string;
}) => {
  const { t } = useTranslation();
  const [name, setName] = useState(initialName);
  const [iconId, setIconId] = useState(initialIconId ?? DEFAULT_LAYOUT_PRESET_ICON_ID);
  const [defaultRoute, setDefaultRoute] = useState(() =>
    getInitialLayoutPresetRoute(initialDefaultRoute, sourceOptions)
  );
  const nameRef = useRef<HTMLInputElement>(null);
  const sourceCollection = useMemo(
    () =>
      createListCollection<SourceSelectItem>({
        items: sourceOptions.map((source) => ({ ...source, value: source.sourceId })),
      }),
    [sourceOptions]
  );
  const sourceTriggerProps = useMemo(() => ({ 'aria-label': t('topbar.presets.defaultSource') }), [t]);
  const selectedSource = sourceOptions.find((source) => source.sourceId === defaultRoute?.sourceId);
  const sourceValue = useMemo(() => (defaultRoute ? [defaultRoute.sourceId] : []), [defaultRoute]);
  const canSubmit = name.trim().length > 0 && (sourceOptions.length === 0 || defaultRoute !== undefined);

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

      onSubmit({ defaultRoute: defaultRoute ?? null, iconId, name: trimmed });
      onClose();
    },
    [defaultRoute, iconId, name, onClose, onSubmit]
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
  const handleSourceChange = useCallback((event: { value: string[] }) => {
    const sourceId = event.value[0] as InvocationSourceId | undefined;

    if (!sourceId) {
      return;
    }

    setDefaultRoute((route) => ({
      destination: route?.destination ?? getNaturalDestination(sourceId),
      sourceId,
    }));
  }, []);
  const handleDestinationChange = useCallback((event: { value: string | null }) => {
    if (!event.value) {
      return;
    }

    setDefaultRoute((route) => (route ? { ...route, destination: event.value as ResultDestination } : route));
  }, []);

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
                  <Field
                    helpText={sourceOptions.length === 0 ? t('topbar.presets.noInvocationSource') : undefined}
                    label={t('topbar.presets.defaultSource')}
                  >
                    <Select
                      collection={sourceCollection}
                      disabled={sourceOptions.length === 0}
                      renderItem={renderSourceOption}
                      size="xs"
                      triggerProps={sourceTriggerProps}
                      value={sourceValue}
                      valueText={
                        selectedSource
                          ? renderSourceOption({ ...selectedSource, value: selectedSource.sourceId })
                          : undefined
                      }
                      onValueChange={handleSourceChange}
                    />
                  </Field>
                  <Field disabled={!defaultRoute} label={t('topbar.presets.defaultDestination')}>
                    <SegmentGroup.Root
                      aria-label={t('topbar.presets.defaultDestination')}
                      disabled={!defaultRoute}
                      size="xs"
                      value={defaultRoute?.destination ?? null}
                      onValueChange={handleDestinationChange}
                    >
                      <SegmentGroup.Indicator />
                      {resultDestinations.map((destination) => (
                        <SegmentGroup.Item key={destination.id} flex="1" justifyContent="center" value={destination.id}>
                          <SegmentGroup.ItemHiddenInput />
                          <SegmentGroup.ItemText display="flex" alignItems="center" gap="1.5">
                            <WidgetIcon
                              boxSize="3.5"
                              icon={getWidgetById(destinationWidgetTypeIds[destination.id])?.manifest.icon}
                            />
                            {getDestinationLabel(destination.id)}
                          </SegmentGroup.ItemText>
                        </SegmentGroup.Item>
                      ))}
                    </SegmentGroup.Root>
                  </Field>
                </Stack>
              </Dialog.Body>
              <Dialog.Footer>
                <Button size="sm" type="button" variant="outline" onClick={onClose}>
                  {t('common.cancel')}
                </Button>
                <Button disabled={!canSubmit} size="sm" type="submit">
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
