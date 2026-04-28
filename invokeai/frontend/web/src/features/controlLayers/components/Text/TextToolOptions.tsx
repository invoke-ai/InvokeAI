import {
  Box,
  ButtonGroup,
  Combobox,
  type ComboboxOption,
  CompositeSlider,
  Flex,
  IconButton,
  NumberInput,
  NumberInputField,
  Popover,
  PopoverAnchor,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Portal,
  Text,
  Tooltip,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { GroupBase } from 'chakra-react-select';
import {
  selectTextAlignment,
  selectTextFontId,
  selectTextFontSize,
  selectTextLineHeight,
  textAlignmentChanged,
  textBoldToggled,
  textFontChanged,
  textFontSizeChanged,
  textItalicToggled,
  textLineHeightChanged,
  textStrikethroughToggled,
  textUnderlineToggled,
} from 'features/controlLayers/store/canvasTextSlice';
import {
  resolveAvailableFont,
  setCustomTextFontStacks,
  TEXT_FONT_STACKS,
  TEXT_MAX_FONT_SIZE,
  TEXT_MIN_FONT_SIZE,
  type TextFontId,
} from 'features/controlLayers/text/textConstants';
import type { FocusEvent, KeyboardEvent, MouseEvent } from 'react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiCaretDownBold,
  PiTextAlignCenterBold,
  PiTextAlignLeftBold,
  PiTextAlignRightBold,
  PiTextBBold,
  PiTextItalicBold,
  PiTextStrikethroughBold,
  PiTextUnderlineBold,
} from 'react-icons/pi';
import { useListUserFontsQuery } from 'services/api/endpoints/utilities';

const formatSliderValue = (value: number) => String(value);
const loadedUserFontFaces = new Set<string>();
const truncateLabel = (value: string, maxLength: number = 36): string => {
  if (value.length <= maxLength) {
    return value;
  }
  return `${value.slice(0, maxLength - 3)}...`;
};

export const TextToolOptions = () => {
  return (
    <Flex alignItems="center" gap={2} minW={0} flexShrink={1} overflow="hidden" data-text-tool-safezone="true">
      <FontSelect />
      <FontSizeControl />
      <LineHeightSelect />
      <FormatControls />
      <AlignmentControls />
    </Flex>
  );
};

const FontSelect = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const fontId = useAppSelector(selectTextFontId);
  const { data: userFonts } = useListUserFontsQuery();
  const userFontsLabel = t('controlLayers.text.customFonts', { defaultValue: 'User Fonts' });
  const builtInFontsLabel = t('controlLayers.text.builtInFonts', { defaultValue: 'Built-in Fonts' });

  useEffect(() => {
    if (!userFonts || userFonts.length === 0) {
      setCustomTextFontStacks([]);
      return;
    }
    const customStacks = userFonts.map((font) => ({
      id: font.id,
      label: font.label,
      stack: `"${font.family}",sans-serif`,
    }));
    setCustomTextFontStacks(customStacks);
  }, [userFonts]);

  useEffect(() => {
    if (!userFonts || userFonts.length === 0 || typeof document === 'undefined' || typeof FontFace === 'undefined') {
      return;
    }
    void Promise.all(
      userFonts.flatMap((font) =>
        font.faces.map(async (face) => {
          const faceKey = `${font.family}|${face.weight}|${face.style}|${face.url}`;
          if (loadedUserFontFaces.has(faceKey)) {
            return;
          }
          try {
            const fontFace = new FontFace(font.family, `url("${face.url}")`, {
              weight: String(face.weight),
              style: face.style,
            });
            await fontFace.load();
            document.fonts.add(fontFace);
            loadedUserFontFaces.add(faceKey);
          } catch {
            // Ignore failures and let browser fallback fonts render.
          }
        })
      )
    );
  }, [userFonts]);

  const options = useMemo(() => {
    const customOptions: ComboboxOption[] = (userFonts ?? []).map((font) => {
      return {
        value: font.id,
        label: truncateLabel(font.label),
      };
    });
    const builtInOptions: ComboboxOption[] = TEXT_FONT_STACKS.map(({ id, label, stack }) => {
      const resolved = resolveAvailableFont(stack);
      const display = truncateLabel(`${label} (${resolved})`);
      return {
        value: id,
        label: display,
      };
    });
    if (customOptions.length === 0) {
      return builtInOptions;
    }
    return [
      {
        label: userFontsLabel,
        options: customOptions,
      },
      { label: builtInFontsLabel, options: builtInOptions },
    ] as GroupBase<ComboboxOption>[];
  }, [builtInFontsLabel, userFonts, userFontsLabel]);
  const selectedOption = useMemo(() => {
    const firstOption = options[0];
    const flattened =
      firstOption && 'options' in firstOption
        ? (options as GroupBase<ComboboxOption>[]).flatMap((group) => group.options)
        : (options as ComboboxOption[]);
    return flattened.find((option) => option.value === fontId) ?? null;
  }, [fontId, options]);
  const handleFontChange = useCallback(
    (option: { value: string } | null) => {
      if (!option) {
        return;
      }
      dispatch(textFontChanged(option.value as TextFontId));
    },
    [dispatch]
  );
  const formatFontGroupLabel = useCallback(
    (group: GroupBase<ComboboxOption>) => {
      const isBuiltInGroup = group.label === builtInFontsLabel;
      return (
        <Flex w="full" flexDir="column" gap={1} py={1}>
          {isBuiltInGroup && <Box borderTopWidth="1px" borderTopColor="base.500" opacity={0.85} />}
          <Text fontSize="xs" fontWeight="semibold" color="base.400" textTransform="uppercase" letterSpacing="0.04em">
            {group.label}
          </Text>
        </Flex>
      );
    },
    [builtInFontsLabel]
  );

  return (
    <Flex w="280px" minW="280px" alignItems="center" gap={2}>
      <Text fontSize="sm" lineHeight="1" whiteSpace="nowrap">
        {t('controlLayers.text.font')}
      </Text>
      <Combobox
        size="sm"
        variant="outline"
        isSearchable={false}
        options={options}
        value={selectedOption}
        onChange={handleFontChange}
        formatGroupLabel={formatFontGroupLabel}
      />
    </Flex>
  );
};

const FontSizeControl = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const fontSize = useAppSelector(selectTextFontSize);

  const [localFontSize, setLocalFontSize] = useState(String(fontSize));

  const marks = useMemo(
    () => [1, 100, 200, 300, 400, 500].filter((value) => value >= TEXT_MIN_FONT_SIZE && value <= TEXT_MAX_FONT_SIZE),
    []
  );

  const handleFontSizeCommit = useCallback(
    (value: number) => {
      const clamped = Math.min(Math.max(value, TEXT_MIN_FONT_SIZE), TEXT_MAX_FONT_SIZE);
      setLocalFontSize(String(clamped));
      dispatch(textFontSizeChanged(clamped));
    },
    [dispatch]
  );

  const onChangeNumberInput = useCallback((valueAsString: string) => {
    setLocalFontSize(valueAsString);
  }, []);

  const onBlur = useCallback(() => {
    const num = parseInt(localFontSize, 10);
    if (isNaN(num)) {
      setLocalFontSize(String(fontSize));
    } else {
      handleFontSizeCommit(num);
    }
  }, [localFontSize, fontSize, handleFontSizeCommit]);

  const onChangeSlider = useCallback(
    (value: number) => {
      handleFontSizeCommit(value);
    },
    [handleFontSizeCommit]
  );

  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter' || e.key === 'Escape') {
        onBlur();
        e.currentTarget.blur();
      }
    },
    [onBlur]
  );

  const onFocusNumberInput = useCallback((e: FocusEvent<HTMLInputElement>) => {
    e.currentTarget.select();
  }, []);

  const onMouseDownNumberInput = useCallback((e: MouseEvent<HTMLInputElement>) => {
    // Ensure re-clicking an already-focused input reselects the value.
    e.preventDefault();
    e.currentTarget.focus();
    e.currentTarget.select();
  }, []);

  useEffect(() => {
    setLocalFontSize(String(fontSize));
  }, [fontSize]);

  return (
    <Flex w="auto" flexShrink={0} alignItems="center" gap={2}>
      <Text fontSize="sm" lineHeight="1" whiteSpace="nowrap">
        {t('controlLayers.text.size')}
      </Text>
      <Flex gap={2} alignItems="center">
        <Tooltip label={t('controlLayers.text.size')}>
          <Box w="80px" minW="80px">
            <Popover>
              <PopoverAnchor>
                <NumberInput
                  variant="outline"
                  display="flex"
                  alignItems="center"
                  min={TEXT_MIN_FONT_SIZE}
                  max={TEXT_MAX_FONT_SIZE}
                  value={localFontSize}
                  onChange={onChangeNumberInput}
                  onBlur={onBlur}
                  clampValueOnBlur={true}
                >
                  <NumberInputField
                    _focusVisible={{ zIndex: 0 }}
                    paddingInlineEnd={7}
                    onKeyDown={onKeyDown}
                    onFocus={onFocusNumberInput}
                    onMouseDown={onMouseDownNumberInput}
                  />
                  <Box position="absolute" right="25px" fontSize="xs" color="base.500" pointerEvents="none">
                    {t('controlLayers.text.px')}
                  </Box>
                  <PopoverTrigger>
                    <IconButton
                      aria-label={t('controlLayers.text.size')}
                      icon={<PiCaretDownBold />}
                      size="sm"
                      variant="link"
                      position="absolute"
                      insetInlineEnd={0}
                      h="full"
                    />
                  </PopoverTrigger>
                </NumberInput>
              </PopoverAnchor>
              <Portal>
                <PopoverContent w={200} pt={0} pb={2} px={4} data-text-tool-safezone="true">
                  <PopoverArrow />
                  <PopoverBody>
                    <CompositeSlider
                      min={TEXT_MIN_FONT_SIZE}
                      max={TEXT_MAX_FONT_SIZE}
                      step={1}
                      value={fontSize}
                      onChange={onChangeSlider}
                      marks={marks}
                      formatValue={formatSliderValue}
                      alwaysShowMarks
                    />
                  </PopoverBody>
                </PopoverContent>
              </Portal>
            </Popover>
          </Box>
        </Tooltip>
      </Flex>
    </Flex>
  );
};

const LineHeightSelect = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const lineHeight = useAppSelector(selectTextLineHeight);
  const lineHeightOptions = useMemo(
    () => [
      { value: '0.75', label: t('controlLayers.text.lineHeightDense') },
      { value: '1.0', label: t('controlLayers.text.lineHeightNormal') },
      { value: '1.25', label: t('controlLayers.text.lineHeightSpacious') },
    ],
    [t]
  );
  const selectedOption =
    lineHeightOptions.find((option) => parseFloat(option.value) === lineHeight) ?? lineHeightOptions[1];
  const handleLineHeightChange = useCallback(
    (option: { value: string } | null) => {
      if (!option) {
        return;
      }
      dispatch(textLineHeightChanged(parseFloat(option.value)));
    },
    [dispatch]
  );

  return (
    <Flex w="160px" minW="160px" alignItems="center" gap={2}>
      <Text fontSize="sm" lineHeight="1" whiteSpace="nowrap">
        {t('controlLayers.text.lineHeight')}
      </Text>
      <Tooltip label={t('controlLayers.text.lineHeight')}>
        <Combobox
          size="sm"
          variant="outline"
          isSearchable={false}
          options={lineHeightOptions}
          value={selectedOption}
          onChange={handleLineHeightChange}
        />
      </Tooltip>
    </Flex>
  );
};

const FormatControls = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isBold = useAppSelector((state) => state.canvasText.bold);
  const isItalic = useAppSelector((state) => state.canvasText.italic);
  const isUnderline = useAppSelector((state) => state.canvasText.underline);
  const isStrikethrough = useAppSelector((state) => state.canvasText.strikethrough);
  const handleBoldToggle = useCallback(() => dispatch(textBoldToggled()), [dispatch]);
  const handleItalicToggle = useCallback(() => dispatch(textItalicToggled()), [dispatch]);
  const handleUnderlineToggle = useCallback(() => dispatch(textUnderlineToggled()), [dispatch]);
  const handleStrikethroughToggle = useCallback(() => dispatch(textStrikethroughToggled()), [dispatch]);

  return (
    <ButtonGroup isAttached variant="outline" flexShrink={0} size="sm">
      <Tooltip label={t('controlLayers.text.bold')}>
        <IconButton
          aria-label={t('controlLayers.text.bold')}
          isChecked={isBold}
          onClick={handleBoldToggle}
          icon={<PiTextBBold />}
          size="sm"
        />
      </Tooltip>
      <Tooltip label={t('controlLayers.text.italic')}>
        <IconButton
          aria-label={t('controlLayers.text.italic')}
          isChecked={isItalic}
          onClick={handleItalicToggle}
          icon={<PiTextItalicBold />}
          size="sm"
        />
      </Tooltip>
      <Tooltip label={t('controlLayers.text.underline')}>
        <IconButton
          aria-label={t('controlLayers.text.underline')}
          isChecked={isUnderline}
          onClick={handleUnderlineToggle}
          icon={<PiTextUnderlineBold />}
          size="sm"
        />
      </Tooltip>
      <Tooltip label={t('controlLayers.text.strikethrough')}>
        <IconButton
          aria-label={t('controlLayers.text.strikethrough')}
          isChecked={isStrikethrough}
          onClick={handleStrikethroughToggle}
          icon={<PiTextStrikethroughBold />}
          size="sm"
        />
      </Tooltip>
    </ButtonGroup>
  );
};

const AlignmentControls = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const alignment = useAppSelector(selectTextAlignment);
  const handleAlignLeft = useCallback(() => dispatch(textAlignmentChanged('left')), [dispatch]);
  const handleAlignCenter = useCallback(() => dispatch(textAlignmentChanged('center')), [dispatch]);
  const handleAlignRight = useCallback(() => dispatch(textAlignmentChanged('right')), [dispatch]);

  return (
    <ButtonGroup isAttached variant="outline" flexShrink={0} size="sm">
      <Tooltip label={t('controlLayers.text.alignLeft')}>
        <IconButton
          aria-label={t('controlLayers.text.alignLeft')}
          isChecked={alignment === 'left'}
          onClick={handleAlignLeft}
          icon={<PiTextAlignLeftBold />}
          size="sm"
        />
      </Tooltip>
      <Tooltip label={t('controlLayers.text.alignCenter')}>
        <IconButton
          aria-label={t('controlLayers.text.alignCenter')}
          isChecked={alignment === 'center'}
          onClick={handleAlignCenter}
          icon={<PiTextAlignCenterBold />}
          size="sm"
        />
      </Tooltip>
      <Tooltip label={t('controlLayers.text.alignRight')}>
        <IconButton
          aria-label={t('controlLayers.text.alignRight')}
          isChecked={alignment === 'right'}
          onClick={handleAlignRight}
          icon={<PiTextAlignRightBold />}
          size="sm"
        />
      </Tooltip>
    </ButtonGroup>
  );
};
