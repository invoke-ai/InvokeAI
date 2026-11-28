import type { GalleryItem, GalleryMarksPatch } from '@features/gallery';
import type { ImageActions } from '@workbench/image-actions';

import { Box, HStack, Kbd, Switch, Text } from '@chakra-ui/react';
import { getGalleryItemMarks, toGalleryItemRef } from '@features/gallery/contracts';
import { GALLERY_CULL_HOTKEY_MODE, getGalleryCullModeValues, type GalleryCullCommandId } from '@features/gallery/cull';
import { GalleryMarksControl } from '@features/gallery/marks';
import { Tooltip } from '@platform/ui/Tooltip';
import { firstPartyHotkeyCatalog } from '@workbench/hotkeys/catalog';
import { formatHotkeyLabel } from '@workbench/hotkeys/keys';
import { applyCustomHotkeys } from '@workbench/hotkeys/resolve';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { PreviewDensity } from './previewDensity';

import { PREVIEW_DOCK_BAR_PROPS } from './previewDockBar';

/** What a cull key did, shown on the stage; `token` restarts the fade for a repeated key. */
export interface PreviewCullHudState {
  announcement: string;
  text: string;
  token: number;
}

/** The control steps down with the Preview so it never overflows the bar. */
const CONTROL_SIZES = { compact: 'md', full: 'lg', minimal: 'sm' } as const satisfies Record<PreviewDensity, string>;

const HINT_GROUPS: readonly { ids: readonly GalleryCullCommandId[]; labelKey: string }[] = [
  {
    ids: ['cull.rate1', 'cull.rate2', 'cull.rate3', 'cull.rate4', 'cull.rate5'],
    labelKey: 'widgets.gallery.cull.hintRate',
  },
  {
    ids: ['cull.labelRed', 'cull.labelYellow', 'cull.labelGreen', 'cull.labelBlue'],
    labelKey: 'widgets.gallery.cull.hintLabel',
  },
  { ids: ['cull.bookmark', 'cull.reject', 'cull.clearFlag'], labelKey: 'widgets.gallery.cull.hintFlag' },
];

const CULL_HOTKEYS = firstPartyHotkeyCatalog.filter((hotkey) => hotkey.mode === GALLERY_CULL_HOTKEY_MODE);

/** "1–5" for a run of digits, else the keys one after another; a key the user disabled is left out. */
const formatHintKeys = (keys: readonly string[]): string => {
  const digits = keys.map(Number);
  const isDigitRun =
    keys.length > 2 &&
    keys.every((key) => /^[0-9]$/.test(key)) &&
    digits.every((digit, index) => index === 0 || digit === digits[index - 1]! + 1);

  return isDigitRun ? `${keys[0]}–${keys.at(-1)}` : keys.map((key) => formatHotkeyLabel(key)).join(' ');
};

const useCullHints = (): { keys: string; label: string }[] => {
  const { t } = useTranslation();
  const customHotkeys = useWorkbenchPreferenceSelector((preferences) => preferences.customHotkeys);

  return useMemo(() => {
    const firstKeys = new Map(
      CULL_HOTKEYS.map((hotkey) => [hotkey.id, applyCustomHotkeys(hotkey, customHotkeys).keys[0]] as const)
    );

    return HINT_GROUPS.map(({ ids, labelKey }) => ({
      keys: formatHintKeys(ids.flatMap((id) => firstKeys.get(id) ?? [])),
      label: t(labelKey),
    })).filter(({ keys }) => keys !== '');
  }, [customHotkeys, t]);
};

export interface PreviewCullBarProps {
  actions: ImageActions;
  autoAdvance: boolean;
  item: GalleryItem;
  position: { index: number; total: number } | null;
}

/**
 * Between the stage and the filmstrip in Cull mode: the marks control, key hints, position and the auto-advance switch.
 * Narrower Preview drops the hints, then the position.
 */
export const PreviewCullBar = ({
  actions,
  autoAdvance,
  density,
  item,
  position,
}: PreviewCullBarProps & { density: PreviewDensity }) => {
  const { t } = useTranslation();
  const { widgets } = useWorkbenchCommands();
  const hints = useCullHints();
  const targets = useMemo(() => [getGalleryItemMarks(item)], [item]);
  const changeMarks = useCallback(
    (patch: GalleryMarksPatch) => void actions.setItemsMarks([toGalleryItemRef(item)], patch),
    [actions, item]
  );
  const setAutoAdvance = useCallback(
    ({ checked }: { checked: boolean }) =>
      widgets.patchValues('gallery', getGalleryCullModeValues({ autoAdvance: checked })),
    [widgets]
  );
  const showsPosition = density !== 'minimal' && position !== null && position.index >= 0;

  return (
    <HStack
      {...PREVIEW_DOCK_BAR_PROPS}
      aria-label={t('widgets.gallery.cull.bar')}
      columnGap="3"
      data-preview-cull-bar=""
      flexWrap="wrap"
      justify="center"
      minH="2.75rem"
      py="1"
      role="group"
      rowGap="1"
    >
      <GalleryMarksControl size={CONTROL_SIZES[density]} targets={targets} onChange={changeMarks} />
      {density === 'full' && hints.length > 0 ? (
        <HStack aria-hidden="true" color="fg.muted" data-preview-cull-hints="" fontSize="sm" gap="2.5">
          {hints.map(({ keys, label }) => (
            <HStack key={label} gap="1">
              <Kbd textTransform="uppercase">{keys}</Kbd>
              <Text as="span">{label}</Text>
            </HStack>
          ))}
        </HStack>
      ) : null}
      {showsPosition ? (
        <Text color="fg.muted" data-preview-cull-position="" fontSize="sm" fontVariantNumeric="tabular-nums">
          {t('widgets.gallery.cull.position', { index: position.index + 1, total: position.total })}
        </Text>
      ) : null}
      <Tooltip content={t('widgets.gallery.cull.autoAdvanceHint')}>
        <Switch.Root checked={autoAdvance} colorPalette="accent" size="sm" onCheckedChange={setAutoAdvance}>
          <Switch.HiddenInput />
          <Switch.Control>
            <Switch.Thumb />
          </Switch.Control>
          <Switch.Label>
            <Text fontSize="sm">{t('widgets.gallery.cull.autoAdvance')}</Text>
          </Switch.Label>
        </Switch.Root>
      </Tooltip>
    </HStack>
  );
};

const HUD_CSS = {
  // Holds, then fades; reduced motion drops the fade but not the hold, so the confirmation still clears itself.
  animationDelay: '700ms',
  animationDuration: 'var(--wb-motion-duration-slow)',
  animationFillMode: 'forwards',
  animationName: 'fade-out',
  '@media (prefers-reduced-motion: reduce)': { animationDuration: '1ms' },
} as const;

/** The stage's confirmation of a cull key; never takes the pointer and never waits on its fade. */
export const PreviewCullHud = ({ hud }: { hud: PreviewCullHudState | null }) =>
  hud ? (
    <Box
      aria-hidden="true"
      data-preview-cull-hud=""
      display="flex"
      inset="0"
      justifyContent="center"
      pointerEvents="none"
      position="absolute"
      alignItems="center"
      zIndex="1"
    >
      <Box
        key={hud.token}
        bg="bg.panel"
        borderColor="border.emphasized"
        borderWidth="1px"
        boxShadow="lg"
        color={hud.text.startsWith('★') ? 'gold.fg' : 'fg'}
        css={HUD_CSS}
        fontSize="4xl"
        fontWeight="700"
        px="4"
        py="2"
        rounded="lg"
      >
        {hud.text}
      </Box>
    </Box>
  ) : null;
