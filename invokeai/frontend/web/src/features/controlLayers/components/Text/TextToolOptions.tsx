import {
  Box,
  ButtonGroup,
  Combobox,
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

const formatSliderValue = (value: number) => String(value);

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
  const options = useMemo(() => {
    return TEXT_FONT_STACKS.map(({ id, label, stack }) => {
      const resolved = resolveAvailableFont(stack);
      return {
        value: id,
        label: `${label} (${resolved})`,
      };
    });
  }, []);
  const selectedOption = options.find((option) => option.value === fontId) ?? null;
  const handleFontChange = useCallback(
    (option: { value: string } | null) => {
      if (!option) {
        return;
      }
      dispatch(textFontChanged(option.value as TextFontId));
    },
    [dispatch]
  );

  return (
    <Flex w="200px" minW="200px" alignItems="center" gap={2}>
      <Text fontSize="sm" lineHeight="1" whiteSpace="nowrap">
        {t('controlLayers.text.font', { defaultValue: 'Font' })}
      </Text>
      <Tooltip label={t('controlLayers.text.font', { defaultValue: 'Font' })}>
        <Combobox
          size="sm"
          variant="outline"
          isSearchable={false}
          options={options}
          value={selectedOption}
          onChange={handleFontChange}
        />
      </Tooltip>
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
        {t('controlLayers.text.size', { defaultValue: 'Size' })}
      </Text>
      <Flex gap={2} alignItems="center">
        <Tooltip label={t('controlLayers.text.size', { defaultValue: 'Size' })}>
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
                      aria-label={t('controlLayers.text.size', { defaultValue: 'Size' })}
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
      { value: '0.75', label: t('controlLayers.text.lineHeightDense', { defaultValue: 'Dense' }) },
      { value: '1.0', label: t('controlLayers.text.lineHeightNormal', { defaultValue: 'Normal' }) },
      { value: '1.25', label: t('controlLayers.text.lineHeightSpacious', { defaultValue: 'Spacious' }) },
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
        {t('controlLayers.text.lineHeight', { defaultValue: 'Spacing' })}
      </Text>
      <Tooltip label={t('controlLayers.text.lineHeight', { defaultValue: 'Spacing' })}>
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
      <Tooltip label={t('controlLayers.text.bold', { defaultValue: 'Bold' })}>
        <IconButton
          aria-label={t('controlLayers.text.bold', { defaultValue: 'Bold' })}
          isChecked={isBold}
          onClick={handleBoldToggle}
          icon={<PiTextBBold />}
          size="sm"
        />
      </Tooltip>
      <Tooltip label={t('controlLayers.text.italic', { defaultValue: 'Italic' })}>
        <IconButton
          aria-label={t('controlLayers.text.italic', { defaultValue: 'Italic' })}
          isChecked={isItalic}
          onClick={handleItalicToggle}
          icon={<PiTextItalicBold />}
          size="sm"
        />
      </Tooltip>
      <Tooltip label={t('controlLayers.text.underline', { defaultValue: 'Underline' })}>
        <IconButton
          aria-label={t('controlLayers.text.underline', { defaultValue: 'Underline' })}
          isChecked={isUnderline}
          onClick={handleUnderlineToggle}
          icon={<PiTextUnderlineBold />}
          size="sm"
        />
      </Tooltip>
      <Tooltip label={t('controlLayers.text.strikethrough', { defaultValue: 'Strikethrough' })}>
        <IconButton
          aria-label={t('controlLayers.text.strikethrough', { defaultValue: 'Strikethrough' })}
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
      <Tooltip label={t('controlLayers.text.alignLeft', { defaultValue: 'Align Left' })}>
        <IconButton
          aria-label={t('controlLayers.text.alignLeft', { defaultValue: 'Align Left' })}
          isChecked={alignment === 'left'}
          onClick={handleAlignLeft}
          icon={<PiTextAlignLeftBold />}
          size="sm"
        />
      </Tooltip>
      <Tooltip label={t('controlLayers.text.alignCenter', { defaultValue: 'Align Center' })}>
        <IconButton
          aria-label={t('controlLayers.text.alignCenter', { defaultValue: 'Align Center' })}
          isChecked={alignment === 'center'}
          onClick={handleAlignCenter}
          icon={<PiTextAlignCenterBold />}
          size="sm"
        />
      </Tooltip>
      <Tooltip label={t('controlLayers.text.alignRight', { defaultValue: 'Align Right' })}>
        <IconButton
          aria-label={t('controlLayers.text.alignRight', { defaultValue: 'Align Right' })}
          isChecked={alignment === 'right'}
          onClick={handleAlignRight}
          icon={<PiTextAlignRightBold />}
          size="sm"
        />
      </Tooltip>
    </ButtonGroup>
  );
};
