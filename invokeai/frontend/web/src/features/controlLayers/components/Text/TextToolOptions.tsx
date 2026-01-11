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
  textAlignmentChanged,
  textBoldToggled,
  textFontChanged,
  textFontSizeChanged,
  textItalicToggled,
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

const formatPx = (value: number | string) => {
  if (isNaN(Number(value))) {
    return '';
  }
  return `${value} px`;
};

const formatSliderValue = (value: number) => String(value);

export const TextToolOptions = () => {
  return (
    <Flex alignItems="center" gap={2} minW={0} flexShrink={1} overflow="hidden" data-text-tool-safezone="true">
      <FontSelect />
      <FontSizeControl />
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
    <Flex minW={48} display="flex" alignItems="center" gap={2} maxW={64}>
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
  const [localFontSize, setLocalFontSize] = useState(fontSize);
  const marks = useMemo(
    () =>
      [5, 50, 100, 200, 300, 400, 500].filter((value) => value >= TEXT_MIN_FONT_SIZE && value <= TEXT_MAX_FONT_SIZE),
    []
  );
  const onChangeNumberInput = useCallback(
    (valueAsString: string, valueAsNumber: number) => {
      setLocalFontSize(valueAsNumber);
      if (!isNaN(valueAsNumber)) {
        dispatch(textFontSizeChanged(valueAsNumber));
      }
    },
    [dispatch]
  );
  const onChangeSlider = useCallback(
    (value: number) => {
      setLocalFontSize(value);
      dispatch(textFontSizeChanged(value));
    },
    [dispatch]
  );
  const onBlur = useCallback(() => {
    if (isNaN(Number(localFontSize))) {
      setLocalFontSize(fontSize);
      return;
    }
    dispatch(textFontSizeChanged(localFontSize));
  }, [dispatch, fontSize, localFontSize]);

  useEffect(() => {
    setLocalFontSize(fontSize);
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
                  step={1}
                  value={localFontSize}
                  onChange={onChangeNumberInput}
                  onBlur={onBlur}
                  format={formatPx}
                  clampValueOnBlur={false}
                >
                  <NumberInputField _focusVisible={{ zIndex: 0 }} title="" paddingInlineEnd={7} />
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
                      step={2}
                      value={localFontSize}
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
