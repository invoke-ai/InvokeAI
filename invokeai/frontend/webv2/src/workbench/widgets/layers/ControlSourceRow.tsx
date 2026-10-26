import type { ControlImageState } from '@workbench/canvasLayerContent';
import type { ChangeEvent } from 'react';

import { Box, HStack, Icon, Image, Input, Menu, Portal, Text } from '@chakra-ui/react';
import { galleryImageUrls } from '@features/gallery/utility';
import { focusIfUnclaimed } from '@platform/react/focusIfUnclaimed';
import { Button } from '@platform/ui/Button';
import { MenuContent, MenuItem } from '@platform/ui/Menu';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { PropertyControlRow } from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { BrushIcon, ChevronDownIcon, ScanIcon, UploadIcon } from 'lucide-react';
import { useCallback, useId, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import type { ControlSourceAction } from './useControlSourceActions';

const REPLACE_MENU_POSITIONING = { placement: 'bottom-end' } as const;
/** In a narrow pane the three empty-state actions keep their words and drop their icons. */
const NARROW_HIDDEN = { '@container (max-width: 15rem)': { display: 'none' } } as const;

const ACTIONS: readonly { action: ControlSourceAction; icon: typeof UploadIcon }[] = [
  { action: 'upload', icon: UploadIcon },
  { action: 'frame', icon: ScanIcon },
  { action: 'draw', icon: BrushIcon },
];

export interface ControlSourceRowProps {
  state: ControlImageState;
  /** The image the layer derives from, for its thumbnail; null for drawn pixels. */
  imageName: string | null;
  /** What the filled row calls its image. */
  label: string;
  busy: ControlSourceAction | null;
  disabled: boolean;
  onUpload(file: File): void;
  onFrame(): void;
  onDraw(): void;
}

/**
 * The Image row: Upload / Frame / Draw while the layer is empty, else a thumbnail, its name and a Replace menu with
 * the same three actions. Focus that sat on the part that went away moves to the part that replaced it.
 */
export const ControlSourceRow = ({
  busy,
  disabled,
  imageName,
  label,
  onFrame,
  onDraw,
  onUpload,
  state,
}: ControlSourceRowProps) => {
  const { t } = useTranslation();
  const labelId = useId();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const replaceIds = useTooltipTriggerIds();
  // Which part this row showed last, so focus is handed over only when one replaces the other, never on mount.
  const shownRef = useRef<'empty' | 'filled' | null>(null);
  const handOver = useCallback((element: HTMLElement | null, part: 'empty' | 'filled') => {
    if (!element) {
      return;
    }
    if (shownRef.current !== null && shownRef.current !== part) {
      focusIfUnclaimed(element);
    }
    shownRef.current = part;
  }, []);
  const emptyRef = useCallback((element: HTMLElement | null) => handOver(element, 'empty'), [handOver]);
  const filledRef = useCallback((element: HTMLElement | null) => handOver(element, 'filled'), [handOver]);

  const openUpload = useCallback(() => fileInputRef.current?.click(), []);
  const handleFileChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.currentTarget.files?.[0];
      event.currentTarget.value = '';
      if (file) {
        onUpload(file);
      }
    },
    [onUpload]
  );
  const run = useCallback(
    (action: ControlSourceAction) => {
      switch (action) {
        case 'upload':
          openUpload();
          return;
        case 'frame':
          onFrame();
          return;
        case 'draw':
          onDraw();
      }
    },
    [onDraw, onFrame, openUpload]
  );
  const handleMenuSelect = useCallback(({ value }: { value: string }) => run(value as ControlSourceAction), [run]);

  return (
    <PropertyControlRow label={t('widgets.layers.control.rows.image')} labelId={labelId}>
      {state === 'empty' ? (
        <HStack
          ref={emptyRef}
          aria-labelledby={labelId}
          containerType="inline-size"
          gap="1"
          gridColumn="2 / -1"
          minW="0"
          role="group"
        >
          {ACTIONS.map(({ action, icon }) => (
            <SourceActionButton key={action} action={action} busy={busy} disabled={disabled} icon={icon} onRun={run} />
          ))}
        </HStack>
      ) : (
        <HStack ref={filledRef} aria-labelledby={labelId} gap="2" gridColumn="2 / -1" minW="0" role="group">
          {imageName ? (
            <Image
              alt=""
              bg="bg.muted"
              boxSize="7"
              draggable={false}
              flexShrink={0}
              objectFit="cover"
              rounded="sm"
              src={galleryImageUrls.thumbnail(imageName)}
            />
          ) : (
            <Box
              alignItems="center"
              bg="bg.muted"
              boxSize="7"
              color="fg.muted"
              display="flex"
              flexShrink={0}
              justifyContent="center"
              rounded="sm"
            >
              <Icon as={BrushIcon} boxSize="3.5" />
            </Box>
          )}
          <Text flex="1" minW="0" truncate>
            {label}
          </Text>
          <Menu.Root ids={replaceIds} positioning={REPLACE_MENU_POSITIONING} onSelect={handleMenuSelect}>
            <Tooltip content={t('widgets.gallery.picker.replaceImage')} ids={replaceIds}>
              <Menu.Trigger asChild>
                <Button
                  aria-label={t('widgets.layers.control.source.replace')}
                  disabled={disabled}
                  flexShrink={0}
                  loading={busy !== null}
                  size="sm"
                  variant="ghost"
                >
                  {t('widgets.layers.control.source.replace')}
                  <Icon as={ChevronDownIcon} boxSize="3" />
                </Button>
              </Menu.Trigger>
            </Tooltip>
            <Portal>
              <Menu.Positioner>
                <MenuContent>
                  {ACTIONS.map(({ action, icon }) => (
                    <MenuItem key={action} value={action}>
                      <Icon as={icon} boxSize="3.5" />
                      {t(`widgets.layers.control.source.${action}`)}
                    </MenuItem>
                  ))}
                </MenuContent>
              </Menu.Positioner>
            </Portal>
          </Menu.Root>
        </HStack>
      )}
      <Input ref={fileInputRef} accept="image/*" display="none" type="file" onChange={handleFileChange} />
    </PropertyControlRow>
  );
};

const SourceActionButton = ({
  action,
  busy,
  disabled,
  icon,
  onRun,
}: {
  action: ControlSourceAction;
  busy: ControlSourceAction | null;
  disabled: boolean;
  icon: typeof UploadIcon;
  onRun(action: ControlSourceAction): void;
}) => {
  const { t } = useTranslation();
  const handleClick = useCallback(() => onRun(action), [action, onRun]);
  const label = t(`widgets.layers.control.source.${action}`);
  return (
    <Tooltip content={t(`widgets.layers.control.source.${action}Hint`)}>
      {/* Named explicitly: while loading, the spinner replaces the visible text. */}
      <Button
        aria-label={label}
        disabled={disabled || (busy !== null && busy !== action)}
        flex="1"
        loading={busy === action}
        minW="0"
        size="sm"
        variant="outline"
        onClick={handleClick}
      >
        <Icon as={icon} boxSize="3" css={NARROW_HIDDEN} />
        <Text truncate>{label}</Text>
      </Button>
    </Tooltip>
  );
};
