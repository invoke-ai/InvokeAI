import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Button, Combobox, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import type { CanvasEntityAdapter } from 'features/controlLayers/konva/CanvasEntity/types';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsOutBold } from 'react-icons/pi';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';
import { z } from 'zod';

const zMode = z.enum(['fill', 'contain', 'cover']);
type Mode = z.infer<typeof zMode>;

export const TransformFitToBboxButtons = memo(({ adapter }: { adapter: CanvasEntityAdapter }) => {
  const { t } = useTranslation();
  const [mode, setMode] = useState<Mode>('contain');
  const isProcessing = useStore(adapter.transformer.$isProcessing);
  const onClick = useCallback(() => {
    if (mode === 'contain') {
      adapter.transformer.fitToBboxContain();
      return;
    }

    if (mode === 'cover') {
      adapter.transformer.fitToBboxCover();
      return;
    }

    if (mode === 'fill') {
      adapter.transformer.fitToBboxFill();
      return;
    }

    assert<Equals<typeof mode, never>>(false);
  }, [adapter.transformer, mode]);

  const options = useMemo(() => {
    return [
      { value: 'contain', label: t('controlLayers.transform.fitModeContain') },
      { value: 'cover', label: t('controlLayers.transform.fitModeCover') },
      { value: 'fill', label: t('controlLayers.transform.fitModeFill') },
    ];
  }, [t]);

  const value = useMemo(() => options.find((o) => o.value === mode) ?? null, [options, mode]);

  const onChange = useCallback<ComboboxOnChange>((v) => {
    if (!v) {
      return;
    }

    setMode(zMode.parse(v.value));
  }, []);

  return (
    <Flex gap={4} w="full">
      <FormControl maxW={64}>
        <FormLabel m={0}>{t('controlLayers.transform.fitMode')}</FormLabel>
        <Combobox options={options} value={value} onChange={onChange} isSearchable={false} isClearable={false} />
      </FormControl>
      <Button
        leftIcon={<PiArrowsOutBold />}
        size="sm"
        onClick={onClick}
        isLoading={isProcessing}
        loadingText={t('controlLayers.transform.fitToBbox')}
        variant="ghost"
      >
        {t('controlLayers.transform.fitToBbox')}
      </Button>
    </Flex>
  );
});

TransformFitToBboxButtons.displayName = 'TransformFitToBboxButtons';
