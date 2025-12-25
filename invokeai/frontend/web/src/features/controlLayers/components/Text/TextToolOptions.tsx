import {
  Box,
  ButtonGroup,
  Combobox,
  CompositeNumberInput,
  CompositeSlider,
  Flex,
  IconButton,
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
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiTextAlignCenterBold,
  PiTextAlignLeftBold,
  PiTextAlignRightBold,
  PiTextBBold,
  PiTextItalicBold,
  PiTextStrikethroughBold,
  PiTextUnderlineBold,
} from 'react-icons/pi';

const formatPx = (value: number | string) => `${value} px`;

export const TextToolOptions = () => {
  return (
    <Flex alignItems="center" gap={2} minW={0} data-text-tool-safezone="true">
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
  const handleFontSizeChange = useCallback(
    (value: number) => {
      dispatch(textFontSizeChanged(value));
    },
    [dispatch]
  );

  return (
    <Flex w="auto" flexShrink={0} alignItems="center" gap={2}>
      <Text fontSize="sm" lineHeight="1" whiteSpace="nowrap">
        {t('controlLayers.text.size', { defaultValue: 'Size' })}
      </Text>
      <Flex gap={2} alignItems="center">
        <Tooltip label={t('controlLayers.text.size', { defaultValue: 'Size' })}>
          <Box w="80px" minW="80px">
            <CompositeNumberInput
              size="sm"
              variant="outline"
              min={TEXT_MIN_FONT_SIZE}
              max={TEXT_MAX_FONT_SIZE}
              step={1}
              value={fontSize}
              onChange={handleFontSizeChange}
              format={formatPx}
            />
          </Box>
        </Tooltip>
        <Tooltip label={t('controlLayers.text.size', { defaultValue: 'Size' })}>
          <Box w="140px" minW="120px">
            <CompositeSlider
              size="sm"
              variant="outline"
              min={TEXT_MIN_FONT_SIZE}
              max={TEXT_MAX_FONT_SIZE}
              step={2}
              value={fontSize}
              onChange={handleFontSizeChange}
            />
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
